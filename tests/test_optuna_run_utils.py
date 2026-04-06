from datetime import datetime, timezone
from pathlib import Path

import pytest

from optuna_run_utils import (
    make_timestamped_artifact_path,
    make_utc_run_timestamp,
    resolve_existing_study_name,
    resolve_run_study_name,
    sanitize_run_name,
)


def test_make_utc_run_timestamp_normalizes_to_utc():
    timestamp = make_utc_run_timestamp(
        datetime(2026, 4, 6, 21, 15, 9, tzinfo=timezone.utc)
    )

    assert timestamp == "20260406_211509"


def test_sanitize_run_name_replaces_unsupported_characters():
    assert sanitize_run_name("  fit volume/profile  ", default="fallback") == (
        "fit_volume_profile"
    )


def test_resolve_run_study_name_auto_generates_timestamped_name():
    resolved = resolve_run_study_name(
        None,
        default_prefix="lgbm generic logloss mean/std",
        timestamp="20260406_211509",
    )

    assert resolved == {
        "study_name": "lgbm_generic_logloss_mean_std_20260406_211509",
        "study_name_source": "auto",
        "run_timestamp": "20260406_211509",
    }


def test_resolve_run_study_name_preserves_configured_name():
    resolved = resolve_run_study_name(
        "existing-study-name",
        default_prefix="ignored",
        timestamp="20260406_211509",
    )

    assert resolved == {
        "study_name": "existing-study-name",
        "study_name_source": "configured",
        "run_timestamp": "20260406_211509",
    }


def test_resolve_existing_study_name_uses_first_non_empty_candidate():
    assert (
        resolve_existing_study_name(None, "  ", "named-study", setting_name="RECHECK")
        == "named-study"
    )


def test_resolve_existing_study_name_raises_when_missing():
    with pytest.raises(ValueError, match="RECHECK_STUDY_NAME"):
        resolve_existing_study_name("", None, setting_name="RECHECK_STUDY_NAME")


def test_make_timestamped_artifact_path_appends_timestamp_and_suffix():
    path = make_timestamped_artifact_path(
        Path("data/optuna/lgbm"),
        stem="lgbm generic optuna best mean/std",
        suffix=".json",
        timestamp="20260406_211509",
    )

    assert path == Path(
        "data/optuna/lgbm/lgbm_generic_optuna_best_mean_std_20260406_211509.json"
    )
