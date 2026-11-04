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


def _checkpoint(path: Path) -> dict:
    header = json.loads(path.read_text(encoding="utf-8"))
    return load_checkpoint(path, identity_sha256=header["identity_sha256"])


def _semantic_digest(value: dict) -> str:
    payload = dict(value)
    payload.pop("peak_rss_bytes", None)
    return state_digest(payload)


def main() -> int:
    continuous_checkpoint = _checkpoint(RUNS / "continuous/checkpoints/native_book_replay.json")
    profiled_continuous_checkpoint = _checkpoint(RUNS / "profiled_continuous/checkpoints/native_book_replay.json")
    profiled_resume_checkpoint = _checkpoint(RUNS / "profiled_resume/runtime/checkpoints/native_book_replay.json")
    for name, value in (
        ("profiled continuous", profiled_continuous_checkpoint),
        ("profiled resume", profiled_resume_checkpoint),
    ):
        if _semantic_digest(continuous_checkpoint) != _semantic_digest(value):
            raise AssertionError(f"{name} terminal checkpoint differs from the original continuous control")
    products = {}
    for name in ("t60", "t59", "source_quality"):
        base = pd.read_pickle(RUNS / f"continuous/products/{name}.pkl")
        profile_cont = pd.read_pickle(RUNS / f"profiled_continuous/products/{name}.pkl")
        profile_resume = pd.read_pickle(RUNS / f"profiled_resume/products/{name}.pkl")
        assert_frame_equal(base, profile_cont, check_exact=True, check_dtype=True, check_like=False)
        assert_frame_equal(base, profile_resume, check_exact=True, check_dtype=True, check_like=False)
        products[name] = {"exact_against_original_continuous": True, "rows": len(base), "columns": len(base.columns)}
    result = {
        "status": "PASS_PROFILED_OUTPUT_EQUIVALENCE",
        "continuous_terminal_semantic_sha256": _semantic_digest(continuous_checkpoint),
        "profiled_continuous_terminal_semantic_sha256": _semantic_digest(profiled_continuous_checkpoint),
        "profiled_resume_terminal_semantic_sha256": _semantic_digest(profiled_resume_checkpoint),
        "products": products,
        "only_checkpoint_exclusion": "peak_rss_bytes; integrity_sha256 is derived by the production serializer",
        "exit_code": 0,
    }
    out = CONTROL / "resume_profile_compare.json"
    log = CONTROL / "resume_profile_compare.log"
    code = CONTROL / "resume_profile_compare.exit_code"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    code.write_text("0\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log = Path(__file__).resolve().parent / "resume_profile_compare.log"
        code = Path(__file__).resolve().parent / "resume_profile_compare.exit_code"
        log.write_text(f"{type(exc).__name__}: {exc}\nexit_code=1\n", encoding="utf-8")
        code.write_text("1\n", encoding="utf-8")
        raise
