"""Resume the frozen model update after filling its missing T-60 calendar field."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
CONFIG_SHA256 = "15d7ca80ba138e6862c2b96357f4265a3910022c9cf20994237dee94983fc1d1"
DEVELOPMENT_FOLDS = ("dev_2026q2_a", "dev_2026q2_b", "dev_2026q2_c")
VARIANT_NAMES = ("A_reoptimized", "B_dynamic_price_volume")


def _preflight(update, research):
    update._validate_frozen_config()
    if update._sha256(update.CONFIG_PATH) != CONFIG_SHA256:
        raise RuntimeError("Pre-registered model-update configuration changed")
    source_identity = update._source_inputs()[-1]
    manifest = json.loads(update.MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("source_identity") != source_identity:
        raise RuntimeError("Model-update source identity changed during the interrupted run")
    if manifest.get("status") != "running":
        raise RuntimeError(f"Expected interrupted running manifest, found {manifest.get('status')!r}")
    if manifest.get("folds", {}).get("evaluated_holdout"):
        raise RuntimeError("Holdout prediction generation already started; refusing to replay it")

    calendar_columns = set(pq.read_schema(research.BASE_MARKET_CALENDAR_PATH).names)
    if "nominal_decision_at" in calendar_columns:
        raise RuntimeError("The base calendar schema changed; this resume shim is no longer applicable")

    validated_files = []
    for fold_id in DEVELOPMENT_FOLDS:
        for variant_name in VARIANT_NAMES:
            record = manifest.get("folds", {}).get(fold_id, {}).get(variant_name, {})
            path = update.ROOT / record.get("prediction_path", "__missing__")
            if (
                record.get("status") != "completed"
                or not path.is_file()
                or update._sha256(path) != record.get("prediction_sha256")
            ):
                raise RuntimeError(f"Missing or changed development prediction {variant_name}/{fold_id}")
            frame = pd.read_parquet(path, columns=["market_start_utc", "nominal_decision_at"])
            starts = pd.to_datetime(frame["market_start_utc"], utc=True, errors="raise")
            decisions = pd.to_datetime(frame["nominal_decision_at"], utc=True, errors="raise")
            if not starts.eq(decisions + pd.Timedelta(minutes=1)).all():
                raise RuntimeError(f"Development predictions violate the frozen T-60 decision time: {variant_name}/{fold_id}")
            validated_files.append(record["prediction_path"])

    return manifest, source_identity, validated_files


def main() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import run_btc_preopen_model_update as update
    import run_btc_preopen_research as research

    manifest, source_identity, validated_files = _preflight(update, research)
    helper_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["runtime_resume_patch"] = {
        "name": "derive missing evaluation-calendar nominal_decision_at in memory",
        "derived_from": "market_start_utc minus one minute",
        "validation": "all six completed development OOF files satisfy market_start_utc = nominal_decision_at + 1 minute",
        "calendar_schema_missing_column": "nominal_decision_at",
        "model_training_or_prediction_changed": False,
        "pre_registered_configuration_sha256": CONFIG_SHA256,
        "source_identity": source_identity,
        "validated_development_prediction_paths": validated_files,
        "resume_helper_sha256": helper_sha256,
    }
    update._write_json(update.MANIFEST_PATH, manifest)

    original_merge = update._merge_predictions

    def merge_with_t60_decision_time(base_data, candidate_predictions):
        if "nominal_decision_at" in base_data.columns:
            return original_merge(base_data, candidate_predictions)
        enriched = base_data.copy()
        starts = pd.to_datetime(enriched["market_start_utc"], utc=True, errors="raise")
        enriched["nominal_decision_at"] = starts - pd.Timedelta(minutes=1)
        return original_merge(enriched, candidate_predictions)

    update._merge_predictions = merge_with_t60_decision_time
    try:
        result = update.run_model_update()
    finally:
        update._merge_predictions = original_merge

    completed = json.loads(update.MANIFEST_PATH.read_text(encoding="utf-8"))
    if completed.get("status") != "completed":
        raise RuntimeError("Model update did not reach its completed manifest state")
    if completed.get("runtime_resume_patch", {}).get("resume_helper_sha256") != helper_sha256:
        raise RuntimeError("Resume provenance was not preserved in the completed manifest")
    if set(completed.get("folds", {}).get("evaluated_holdout", {})) != set(VARIANT_NAMES):
        raise RuntimeError("Both frozen variants did not complete evaluated_holdout")
    print(
        "[btc-model-update-resume] completed "
        f"development={completed.get('development_freeze', {}).get('common_rows')} "
        f"holdout variants={len(completed['folds']['evaluated_holdout'])}; "
        f"summary={result.get('holdout_rows_common')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
