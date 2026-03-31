import csv
import math
import threading
from pathlib import Path

EXECUTION_SNAPSHOTS_DIR = Path("data/live/execution_snapshots")

# Execution snapshots store one lightweight row per live decision so later
# simulations can replay decision conditions and observed quotes without
# persisting a full orderbook stream.
EXECUTION_SNAPSHOT_COLUMNS = (
    "logged_at_utc",
    "run_started_at_utc",
    "decision_opened",
    "bucket_start",
    "bucket_end",
    "series_slug",
    "market_slug",
    "up_token_id",
    "down_token_id",
    "proba_up_raw",
    "proba_up_kelly_input",
    "up_ask_price",
    "up_ask_size",
    "down_ask_price",
    "down_ask_size",
    "order_price_cap",
    "selected_side",
    "selected_edge",
    "selected_fraction",
    "stake_usdc_intended",
    "trade_allowed",
    "submission_attempted",
    "submission_success",
    "submitted_price",
    "filled_price",
    "execution_mode",
)

_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS = {}


def build_execution_snapshots_path(run_started_at_utc, base_dir=EXECUTION_SNAPSHOTS_DIR):
    return Path(base_dir) / f"execution_snapshots_{str(run_started_at_utc)}.csv"


def _lock_for_path(path):
    path_key = str(Path(path).resolve())
    with _PATH_LOCKS_GUARD:
        if path_key not in _PATH_LOCKS:
            _PATH_LOCKS[path_key] = threading.Lock()
        return _PATH_LOCKS[path_key]


def _serialize_snapshot_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not math.isfinite(value) else value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    return value


def append_execution_snapshot(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized_row = {
        col: _serialize_snapshot_value(row.get(col))
        for col in EXECUTION_SNAPSHOT_COLUMNS
    }

    with _lock_for_path(path):
        write_header = (not path.exists()) or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=EXECUTION_SNAPSHOT_COLUMNS,
                extrasaction="ignore",
            )
            if write_header:
                writer.writeheader()
            writer.writerow(serialized_row)
