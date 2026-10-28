"""Small schema helpers shared by BTC D bundle packaging and verification."""
from __future__ import annotations

import hashlib
from pathlib import Path


def calibration_manifest_fields(
    calibrator: dict | None,
    calibration_status: dict,
    minimum_rows: int,
) -> dict:
    if isinstance(minimum_rows, bool) or not isinstance(minimum_rows, int) or minimum_rows < 1:
        raise RuntimeError("Frozen calibration minimum_rows must be a positive integer")
    if not isinstance(calibration_status, dict) or "status" not in calibration_status:
        raise RuntimeError("Calibration status metadata is missing its status")

    status = calibration_status["status"]
    if status == "fitted":
        if not isinstance(calibrator, dict) or calibrator.get("status") != "fitted":
            raise RuntimeError("Fitted calibration status has no fitted calibrator record")
        if "training_rows" not in calibrator:
            raise RuntimeError("Fitted calibrator is missing training_rows")
        rows = calibrator["training_rows"]
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < minimum_rows:
            raise RuntimeError("Fitted calibrator training_rows conflicts with the frozen minimum")
        if "rows" in calibration_status and calibration_status["rows"] != rows:
            raise RuntimeError("Fitted calibration status rows conflicts with training_rows")
        if "required" in calibration_status and calibration_status["required"] != minimum_rows:
            raise RuntimeError("Fitted calibration status required minimum conflicts with protocol")
        if "training_rows" in calibration_status and calibration_status["training_rows"] != rows:
            raise RuntimeError("Fitted calibration status training_rows conflicts with calibrator")
        return {
            "status": "fitted",
            "rows": rows,
            "required_rows": minimum_rows,
            "calibrator": calibrator,
        }

    if status not in {"minimum_not_met", "both_classes_not_present"}:
        raise RuntimeError(f"Unknown official calibration status: {status!r}")
    if calibrator is not None:
        raise RuntimeError("Unavailable calibration status must not contain a calibrator")
    if "rows" not in calibration_status or "required" not in calibration_status:
        raise RuntimeError("Unavailable calibration status is missing rows or required")
    rows = calibration_status["rows"]
    required = calibration_status["required"]
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise RuntimeError("Unavailable calibration rows must be a nonnegative integer")
    if isinstance(required, bool) or not isinstance(required, int) or required != minimum_rows:
        raise RuntimeError("Unavailable calibration minimum conflicts with the frozen protocol")
    if status == "minimum_not_met" and rows >= required:
        raise RuntimeError("minimum_not_met calibration status contradicts its row count")
    if status == "both_classes_not_present" and rows < required:
        raise RuntimeError("both_classes_not_present status contradicts the frozen row minimum")
    return {
        "status": status,
        "rows": rows,
        "required_rows": required,
        "calibrator": None,
    }


def bundle_status_from_calibration(calibration: dict) -> str:
    if calibration.get("status") == "fitted" and isinstance(calibration.get("calibrator"), dict):
        return "D_BUNDLE_READY"
    return "D_BUNDLE_BLOCKED_CALIBRATION_MISSING"


def verify_file_hash(path: Path, expected_sha256: str, label: str) -> str:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 differs: {path}")
    return actual
