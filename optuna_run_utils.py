import re
from datetime import datetime, timezone
from pathlib import Path


def make_utc_run_timestamp(now=None):
    resolved = datetime.now(timezone.utc) if now is None else now
    if resolved.tzinfo is None:
        resolved = resolved.replace(tzinfo=timezone.utc)
    else:
        resolved = resolved.astimezone(timezone.utc)
    return resolved.strftime("%Y%m%d_%H%M%S")


def sanitize_run_name(value, *, default):
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    sanitized = sanitized.strip("._-")
    return sanitized or default


def resolve_run_study_name(configured_name, *, default_prefix, timestamp=None):
    run_timestamp = timestamp or make_utc_run_timestamp()
    raw_name = str(configured_name or "").strip()
    if raw_name:
        return {
            "study_name": raw_name,
            "study_name_source": "configured",
            "run_timestamp": run_timestamp,
        }
    return {
        "study_name": (
            f"{sanitize_run_name(default_prefix, default='optuna_study')}_{run_timestamp}"
        ),
        "study_name_source": "auto",
        "run_timestamp": run_timestamp,
    }


def resolve_existing_study_name(*candidate_names, setting_name="study_name"):
    for candidate in candidate_names:
        raw_name = str(candidate or "").strip()
        if raw_name:
            return raw_name
    raise ValueError(f"{setting_name} must be set to load an existing Optuna study.")


def make_timestamped_artifact_path(output_dir, *, stem, suffix, timestamp=None):
    run_timestamp = timestamp or make_utc_run_timestamp()
    safe_stem = sanitize_run_name(stem, default="artifact")
    return Path(output_dir) / f"{safe_stem}_{run_timestamp}{suffix}"


__all__ = [
    "make_timestamped_artifact_path",
    "make_utc_run_timestamp",
    "resolve_existing_study_name",
    "resolve_run_study_name",
    "sanitize_run_name",
]
