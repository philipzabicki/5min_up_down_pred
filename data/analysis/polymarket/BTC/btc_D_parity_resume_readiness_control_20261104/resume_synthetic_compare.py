from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from e0037_trace_control import ROOT

sys.path.insert(0, str(ROOT))
from features.btc_D_continuation_20261029 import load_checkpoint, state_digest


CONTROL = Path(__file__).resolve().parent
RUNS = CONTROL / "runs" / "resume_cursor2340"
OUTPUT = CONTROL / "resume_synthetic_compare.json"
LOG = CONTROL / "resume_synthetic_compare.log"
EXIT = CONTROL / "resume_synthetic_compare.exit_code"


def _load(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return load_checkpoint(path, identity_sha256=raw["identity_sha256"])


def _semantic_digest(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("peak_rss_bytes", None)
    return state_digest(semantic)


def main() -> int:
    cursor_continuous_path = RUNS / "continuous/checkpoints/cursor24_checkpoint.json"
    cursor_resumed_path = RUNS / "interrupt/checkpoints/cursor24_checkpoint.json"
    final_continuous_path = RUNS / "continuous/checkpoints/native_book_replay.json"
    final_resumed_path = RUNS / "interrupt/checkpoints/native_book_replay.json"
    cursor_continuous = _load(cursor_continuous_path)
    cursor_resumed = _load(cursor_resumed_path)
    final_continuous = _load(final_continuous_path)
    final_resumed = _load(final_resumed_path)

    if cursor_continuous["hour_cursor"] != 24 or cursor_resumed["hour_cursor"] != 24:
        raise AssertionError("cursor comparison files do not both contain local next-hour cursor 24")
    if cursor_continuous["identity_sha256"] != cursor_resumed["identity_sha256"]:
        raise AssertionError("cursor checkpoint identities differ")
    if _semantic_digest(cursor_continuous) != _semantic_digest(cursor_resumed):
        raise AssertionError("semantic state differs at local cursor 24")
    if final_continuous["hour_cursor"] != 25 or final_resumed["hour_cursor"] != 25:
        raise AssertionError("terminal checkpoint cursor differs from expected 25 selected hours")
    if final_continuous["identity_sha256"] != final_resumed["identity_sha256"]:
        raise AssertionError("terminal checkpoint identities differ")
    if _semantic_digest(final_continuous) != _semantic_digest(final_resumed):
        raise AssertionError("semantic state differs after resumed continuation")

    product_comparisons = {}
    for name in ("t60", "t59", "source_quality"):
        continuous_frame = pd.read_pickle(RUNS / f"continuous/products/{name}.pkl")
        resumed_frame = pd.read_pickle(RUNS / f"resume/products/{name}.pkl")
        assert_frame_equal(continuous_frame, resumed_frame, check_exact=True, check_dtype=True, check_like=False)
        product_comparisons[name] = {"exact": True, "rows": len(continuous_frame), "columns": len(continuous_frame.columns)}

    cursor_counters = {
        key: cursor_continuous[key]
        for key in ("event_parts_read", "event_projected_rows_read", "event_rows_replayed", "i60", "i59")
    }
    terminal_counters = {
        key: final_continuous[key]
        for key in ("event_parts_read", "event_projected_rows_read", "event_rows_replayed", "i60", "i59")
    }
    result = {
        "status": "PASS_SYNTHETIC_SCOPE",
        "comparison_exclusion_declared_before_runs": ["peak_rss_bytes", "derived integrity_sha256"],
        "cursor24": {
            "semantic_digest_continuous": _semantic_digest(cursor_continuous),
            "semantic_digest_resumed": _semantic_digest(cursor_resumed),
            "exact_semantic_equality": True,
            "active_markets": len(cursor_continuous["states"]),
            "active_tokens": sum(len(state.get("tokens", {})) for state in cursor_continuous["states"].values()),
            "t60_captures": len(cursor_continuous["t60"]),
            "t59_captures": len(cursor_continuous["t59"]),
            "peak_rss_continuous_bytes": cursor_continuous["peak_rss_bytes"],
            "peak_rss_interrupted_bytes": cursor_resumed["peak_rss_bytes"],
            "counters": cursor_counters,
        },
        "after_continuation": {
            "semantic_digest_continuous": _semantic_digest(final_continuous),
            "semantic_digest_resumed": _semantic_digest(final_resumed),
            "exact_semantic_equality": True,
            "active_markets": len(final_continuous["states"]),
            "t60_captures": len(final_continuous["t60"]),
            "t59_captures": len(final_continuous["t59"]),
            "peak_rss_continuous_bytes": final_continuous["peak_rss_bytes"],
            "peak_rss_resumed_bytes": final_resumed["peak_rss_bytes"],
            "counters": terminal_counters,
            "products": product_comparisons,
        },
        "authentic_cursor2340_checkpoint_tested": False,
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
        LOG.write_text(f"{type(exc).__name__}: {exc}\nexit_code=1\n", encoding="utf-8")
        EXIT.write_text("1\n", encoding="utf-8")
        raise
