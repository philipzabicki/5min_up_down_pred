"""Resume the frozen model update after correcting the freeze-rule lookup in memory."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import run_btc_preopen_model_update as update
    import run_btc_preopen_research as research
    from scripts.resume_btc_preopen_model_update_after_calendar_keyerror import _preflight

    manifest, source_identity, validated_files = _preflight(update, research)
    rule = update.CONFIG["development_model_freeze"]
    helper_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["runtime_resume_patch_additional"] = {
        "name": "supply the existing freeze rule under the key expected by the frozen serializer",
        "source_issue": "_development_freeze indexes development_model_freeze twice",
        "configuration_rule_sha256": hashlib.sha256(
            json.dumps(rule, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "configuration_or_model_selection_changed": False,
        "source_identity": source_identity,
        "validated_development_prediction_paths": validated_files,
        "resume_helper_sha256": helper_sha256,
    }
    update._write_json(update.MANIFEST_PATH, manifest)

    original_merge = update._merge_predictions
    original_freeze = update._development_freeze

    def merge_with_t60_decision_time(base_data, candidate_predictions):
        if "nominal_decision_at" in base_data.columns:
            return original_merge(base_data, candidate_predictions)
        enriched = base_data.copy()
        starts = pd.to_datetime(enriched["market_start_utc"], utc=True, errors="raise")
        enriched["nominal_decision_at"] = starts - pd.Timedelta(minutes=1)
        return original_merge(enriched, candidate_predictions)

    def freeze_with_existing_rule(dev, baseline_identity, candidate_predictions):
        original_config = update.CONFIG
        existing_rule = original_config["development_model_freeze"]
        augmented_config = dict(original_config)
        augmented_config["development_model_freeze"] = {
            **existing_rule,
            "development_model_freeze": existing_rule,
        }
        update.CONFIG = augmented_config
        try:
            return original_freeze(dev, baseline_identity, candidate_predictions)
        finally:
            update.CONFIG = original_config

    update._merge_predictions = merge_with_t60_decision_time
    update._development_freeze = freeze_with_existing_rule
    try:
        result = update.run_model_update()
    finally:
        update._merge_predictions = original_merge
        update._development_freeze = original_freeze

    completed = json.loads(update.MANIFEST_PATH.read_text(encoding="utf-8"))
    if completed.get("status") != "completed":
        raise RuntimeError("Model update did not reach its completed manifest state")
    if completed.get("runtime_resume_patch_additional", {}).get("resume_helper_sha256") != helper_sha256:
        raise RuntimeError("Freeze-rule resume provenance was not preserved in the completed manifest")
    if set(completed.get("folds", {}).get("evaluated_holdout", {})) != set(update.VARIANTS):
        raise RuntimeError("Both frozen variants did not complete evaluated_holdout")
    print(
        "[btc-model-update-resume] completed "
        f"development={completed.get('development_freeze', {}).get('common_rows')} "
        f"holdout variants={len(completed['folds']['evaluated_holdout'])}; "
        f"holdout rows={result.get('holdout_rows_common')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
