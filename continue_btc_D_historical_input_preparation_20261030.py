"""Resume only the authenticated terminal E0048 v4 book-replay suffix."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "4"

import pandas as pd

from features import btc_D_historical_input_preparation_20261030 as e0048
import run_btc_D_historical_input_preparation_20261030 as runner


RUNTIME = e0048.RUNTIME
CHECKPOINT_PATH = RUNTIME / "checkpoints" / "native_book_replay.json"
EXECUTION_PATH = RUNTIME / "continuation_execution.json"
LOG_PATH = RUNTIME / "continuation_execution.log"
ORIGINAL_RUNNER_PID = 43936
ORIGINAL_RUNNER_NAME = "run_btc_D_historical_input_preparation_20261030.py"
FROZEN_PROTOCOL_SHA256 = "ba79ec4d20787d69058609824514416338db5178a900cd4e9c26b3b84ac8f55a"
FROZEN_FEATURE_MODULE_SHA256 = "00a9961d907b521edcfe7e15fefea079b7b47c82a36609a1b9f4b4924246ed4e"
FROZEN_RUNNER_SHA256 = "a9eb653d5d779a45523c56e8f8d59193874d41652b33132cd254a85ce5157fe3"
START_CURSOR_MIN = 2_340
EXPECTED_HOURS = 2_377
ORIGINAL_RUN_STARTED_UTC = "2026-10-30T01:04:44Z"
WALL_BUDGET_SECONDS = 14_400


def _log(message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_head() -> dict:
    with CHECKPOINT_PATH.open("rb") as stream:
        head = stream.read(8192).decode("utf-8")
    result = {}
    for key in ("hour_cursor", "event_parts_read", "event_projected_rows_read", "event_rows_replayed"):
        match = re.search(r'"' + key + r'":([0-9]+)', head)
        if not match:
            raise RuntimeError(f"native book checkpoint header is missing {key}")
        result[key] = int(match.group(1))
    return result


def _assert_original_runner_stopped() -> None:
    if os.name != "nt":
        raise RuntimeError("E0048 continuation is frozen to the Windows runner environment")
    runner_pattern = re.escape(ORIGINAL_RUNNER_NAME)
    command = (
        f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={ORIGINAL_RUNNER_PID}'; "
        f"if ($null -ne $p -and $p.CommandLine -match '{runner_pattern}') {{ exit 23 }}"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode == 23:
        raise RuntimeError(f"frozen E0048 runner PID {ORIGINAL_RUNNER_PID} is still active")
    if result.returncode != 0:
        raise RuntimeError("could not verify that the original frozen E0048 runner exited")


def _install_part_progress_logger():
    import pyarrow.parquet as pq

    original = pq.ParquetFile
    seen: set[str] = set()

    class ObservedParquetFile:
        def __init__(self, source, *args, **kwargs):
            self._source = source
            self._inner = original(source, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def iter_batches(self, *args, **kwargs):
            name = Path(self._source).name
            if name not in seen:
                seen.add(name)
                _log(f"PMXT_PART_READ_START name={name}")
            return self._inner.iter_batches(*args, **kwargs)

    pq.ParquetFile = ObservedParquetFile
    return pq, original


def _load_attempts() -> list[dict]:
    if not EXECUTION_PATH.is_file():
        return []
    saved = json.loads(EXECUTION_PATH.read_text(encoding="utf-8"))
    if saved.get("protocol_sha256") != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("existing continuation record has a different frozen protocol identity")
    return list(saved.get("attempts", []))


def _run() -> dict:
    _assert_original_runner_stopped()
    protocol, protocol_sha256 = e0048.load_protocol()
    if protocol_sha256 != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("E0048 protocol SHA does not match the published v4 freeze")
    feature_module_sha = e0048.sha256_file(Path(e0048.__file__))
    runner_sha = e0048.sha256_file(Path(runner.__file__))
    if feature_module_sha != FROZEN_FEATURE_MODULE_SHA256 or runner_sha != FROZEN_RUNNER_SHA256:
        raise RuntimeError("frozen E0048 source changed; continuation is refused")
    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError("E0048 native-book checkpoint is missing")

    identity = e0048._feature_identity(
        protocol_sha256,
        protocol["parent"]["bundle_manifest_sha256"],
        protocol["parent"]["checkpoint_semantic_sha256"],
    )
    feature_checkpoint = runner._checkpoint_payload(protocol_sha256, identity)
    if int(feature_checkpoint["processed_feature_rows"]) != e0048.EXPECTED_FEATURE_ROWS:
        raise RuntimeError("frozen E0048 minute-feature checkpoint is incomplete")
    panel_path = RUNTIME / "panel_repeat" / "feature_rows_20260815.parquet"
    if not panel_path.is_file():
        raise RuntimeError("saved deterministic E0048 panel is missing; continuation will not reread it")

    market_rows, _market_sources = e0048._calendar_mapping_inputs(protocol)
    confirmed = market_rows.loc[market_rows.market_identity_status.eq("CONFIRMED")].copy()
    confirmed["market_start_utc"] = pd.to_datetime(confirmed["market_start_utc"], utc=True, errors="raise")
    first_hour = (confirmed.market_start_utc.dt.floor("h") - pd.Timedelta(hours=25)).min()
    last_hour = (confirmed.market_start_utc - pd.Timedelta(seconds=59)).dt.floor("h").max()
    hours = pd.date_range(first_hour, last_hour, freq="h")
    if len(hours) != EXPECTED_HOURS:
        raise RuntimeError(f"frozen market calendar implies {len(hours)} replay hours, expected {EXPECTED_HOURS}")

    before = _checkpoint_head()
    if not START_CURSOR_MIN <= before["hour_cursor"] <= EXPECTED_HOURS:
        raise RuntimeError(f"checkpoint cursor {before['hour_cursor']} is outside the authorized suffix")
    resources_before = runner._resource_gate()
    if EXECUTION_PATH.exists() and not _load_attempts():
        raise RuntimeError("existing continuation record has no auditable attempt history")

    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    attempt = {
        "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
        "status": "RUNNING",
        "checkpoint_before": before,
        "protocol_sha256": protocol_sha256,
        "feature_module_sha256": feature_module_sha,
        "frozen_runner_sha256": runner_sha,
        "remaining_hours_at_start": EXPECTED_HOURS - before["hour_cursor"],
        "original_four_hour_budget_already_exceeded": True,
        "evaluation_authorized": False,
        "holdout_status": "HOLDOUT_NOT_RUN",
    }
    attempts = _load_attempts()
    attempts.append(attempt)
    execution = {
        "schema_version": 1,
        "experiment_id": "E0048",
        "task_id": "T0017",
        "artifact_type": "E0048_CHECKPOINT_CONTINUATION_EXECUTION",
        "status": "RUNNING",
        "protocol_sha256": protocol_sha256,
        "attempts": attempts,
    }
    _atomic_json(EXECUTION_PATH, execution)
    _log(f"CONTINUATION_START cursor={before['hour_cursor']}/{EXPECTED_HOURS} remaining_hours={EXPECTED_HOURS - before['hour_cursor']} available_ram_gib={resources_before['available_ram_bytes'] / 1024**3:.2f} free_disk_gib={resources_before['free_workspace_disk_bytes'] / 1024**3:.2f}")

    try:
        pq, original_parquet_file = _install_part_progress_logger()
        replay_started = time.perf_counter()
        try:
            t60, t59, source_quality, book_manifest = e0048.replay_native_books(protocol, market_rows, protocol_sha256)
        finally:
            pq.ParquetFile = original_parquet_file
        replay_seconds = time.perf_counter() - replay_started
        after_replay = _checkpoint_head()
        if after_replay["hour_cursor"] != EXPECTED_HOURS:
            raise RuntimeError(f"replay ended at {after_replay['hour_cursor']}/{EXPECTED_HOURS} hours")
        event_counter_delta = {
            key: after_replay[key] - before[key]
            for key in ("event_parts_read", "event_projected_rows_read", "event_rows_replayed")
        }
        if before["hour_cursor"] == EXPECTED_HOURS and any(event_counter_delta.values()):
            raise RuntimeError("terminal-checkpoint product recovery unexpectedly read PMXT event payload rows")
        if after_replay["event_rows_replayed"] < before["event_rows_replayed"]:
            raise RuntimeError("replayed-event counter moved backwards")
        _log(f"REPLAY_COMPLETE cursor={after_replay['hour_cursor']}/{EXPECTED_HOURS} parts_delta={after_replay['event_parts_read'] - before['event_parts_read']} projected_rows_delta={after_replay['event_projected_rows_read'] - before['event_projected_rows_read']} replayed_rows_delta={after_replay['event_rows_replayed'] - before['event_rows_replayed']} elapsed_seconds={replay_seconds:.1f}")

        product_started = time.perf_counter()
        features, flow_status, feature_meta = runner._collect_feature_rows(protocol)
        input_rows, coverage, summary, products = runner._build_slot_products(
            protocol, market_rows, features, flow_status, t60, t59, source_quality,
        )
        identity_confirmed = coverage.market_identity_status.eq("CONFIRMED").fillna(False)
        unmatched_identity_slots = int((~identity_confirmed).sum())
        if unmatched_identity_slots != 2:
            raise RuntimeError(f"E0043 metadata-only reconciliation expected two unmatched starts, observed {unmatched_identity_slots}")
        entry_available = coverage.t59_archive_entry_hour_available.fillna(False).astype(bool)
        freshness_available = coverage.t59_archive_freshness_window_available.fillna(False).astype(bool)
        missing_history = pd.to_numeric(coverage.t59_archive_window_missing_source_hours, errors="coerce").fillna(0).gt(0)
        gap_counts = {
            "market_windows_affected_at_t59": int(missing_history.sum()),
            # E0043's gap counts include confirmed identities; the two unmatched starts are separate.
            "windows_without_entry_or_freshness": int(((~entry_available | ~freshness_available) & identity_confirmed & missing_history).sum()),
            "windows_with_entry_but_missing_prior_history": int((entry_available & identity_confirmed & missing_history).sum()),
        }
        if gap_counts != {
            "market_windows_affected_at_t59": 1_115,
            "windows_without_entry_or_freshness": 816,
            "windows_with_entry_but_missing_prior_history": 299,
        }:
            raise RuntimeError(f"E0048 T-59 coverage does not reconcile with frozen E0043 counts: {gap_counts}")
        if len(input_rows) != e0048.EXPECTED_MARKET_SLOTS or len(t60) != e0048.EXPECTED_MARKET_SLOTS or len(t59) != e0048.EXPECTED_MARKET_SLOTS:
            raise RuntimeError("E0048 slot or book product count differs from the frozen market calendar")
        resource_after = runner._resource_gate()
        product_seconds = time.perf_counter() - product_started
        completed_at = datetime.now(timezone.utc)
        elapsed_seconds = time.perf_counter() - started_clock
        attempt.update({
            "status": "SOURCE_PRODUCTS_BUILT_BUDGET_EXCEEDED",
            "completed_at_utc": completed_at.isoformat().replace("+00:00", "Z"),
            "checkpoint_after": after_replay,
            "replay_seconds": replay_seconds,
            "product_build_seconds": product_seconds,
            "continuation_process_elapsed_seconds": elapsed_seconds,
            "cumulative_wall_lower_bound_seconds": (completed_at - datetime.fromisoformat(ORIGINAL_RUN_STARTED_UTC.replace("Z", "+00:00"))).total_seconds(),
            "frozen_wall_budget_seconds": WALL_BUDGET_SECONDS,
            "gap_reconciliation": gap_counts,
            "unmatched_market_identity_slots": unmatched_identity_slots,
            "feature_rows": feature_meta["rows"],
            "slot_rows": len(input_rows),
            "products": products,
            "book_replay": book_manifest,
            "resources_before": resources_before,
            "resources_after": resource_after,
            "evaluation_authorized": False,
            "activated": False,
            "holdout_status": "HOLDOUT_NOT_RUN",
        })
        execution.update({"status": attempt["status"], "created_at_utc": completed_at.isoformat().replace("+00:00", "Z")})
        _atomic_json(EXECUTION_PATH, execution)
        _log(f"PRODUCTS_COMPLETE slots={len(input_rows)} feature_rows={feature_meta['rows']} elapsed_seconds={elapsed_seconds:.1f} status={attempt['status']}")
        return execution
    except BaseException as exc:
        current = _checkpoint_head()
        attempt.update({
            "status": "FAILED_RESUMABLE",
            "failed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "checkpoint_after_failure": current,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        execution.update({"status": "FAILED_RESUMABLE"})
        _atomic_json(EXECUTION_PATH, execution)
        _log(f"CONTINUATION_FAILED_RESUMABLE cursor={current['hour_cursor']}/{EXPECTED_HOURS} error={type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    _run()
