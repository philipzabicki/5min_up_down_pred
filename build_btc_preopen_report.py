"""Assemble the BTC pre-open audit report and its verification bundle."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports/btc_preopen"
RUN_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/nightly_v1/06dc69ccb35a63b3"
EXTRACT_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/pmxt_v2_archive_full_scenarios"
ORIGINAL_TRAINING_MANIFEST = RUN_DIR / "stages/final_model_51d3adab0c53/training_manifest.json"
ORIGINAL_EVALUATION = RUN_DIR / "stages/external_evaluation_7e5807168d4c/evaluation.json"
ORIGINAL_MODEL_TUNING = RUN_DIR / "stages/model_tuning_76c54c344f7b/best_result.json"
ORIGINAL_CONFIG = RUN_DIR / "effective_config.json"
ORIGINAL_RUN_MANIFEST = RUN_DIR / "run_manifest.json"
CANDIDATE_METRICS = REPORT_DIR / "candidate_metrics.json"
ECONOMIC_REPLAY = REPORT_DIR / "economic_replay.json"
ECONOMIC_SCENARIOS = REPORT_DIR / "economic_scenarios.csv"
COVERAGE_CSV = REPORT_DIR / "data_coverage.csv"
MODEL_COMPARISON = REPORT_DIR / "model_comparison.csv"
TRADES = REPORT_DIR / "trades.parquet"
INFERENCE_LATENCY = REPORT_DIR / "candidate_inference_latency.json"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not pd.notna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _get_scenario(economic, entry_case, age, release_delay):
    rows = economic.loc[
        economic.entry_case.eq(entry_case)
        & economic.max_ask_age_seconds.eq(age)
        & economic.settlement_release_delay_seconds.eq(release_delay)
    ]
    return rows.sort_values("model").to_dict("records")


def _money(value):
    return "-" if pd.isna(value) else f"${float(value):.2f}"


def _repair_report_line(value):
    if not any(marker in value for marker in ("\u0139", "\u00c4", "\u0102", "\u00e2")):
        return value
    raw = bytearray()
    try:
        for character in value:
            codepoint = ord(character)
            if 0x80 <= codepoint <= 0x9F:
                raw.append(codepoint)
            else:
                raw.extend(character.encode("cp1250"))
        return raw.decode("utf-8")
    except UnicodeError:
        return value

def _coverage_counts(coverage, case, age):
    return coverage.loc[coverage.entry_case.eq(case), f"coverage_reason_age{age}s"].value_counts().to_dict()


def build():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run = _json(ORIGINAL_RUN_MANIFEST)
    config = _json(ORIGINAL_CONFIG)
    training = _json(ORIGINAL_TRAINING_MANIFEST)
    evaluation = _json(ORIGINAL_EVALUATION)
    tuning = _json(ORIGINAL_MODEL_TUNING)
    indicator_stage = _json(RUN_DIR / "stages/indicators_b150948edf4d/indicator_stage.json")
    weight_selection = _json(RUN_DIR / "stages/target_weights_7f8808f29356/target_weight_result.json")
    volume_selection = _json(RUN_DIR / "stages/volume_profile_d9bd4bb26193/best_result.json")
    reaction_selection = _json(RUN_DIR / "stages/reaction_profile_47fc0f8c0603/best_result.json")
    candidate = _json(CANDIDATE_METRICS)
    inference_latency = _json(INFERENCE_LATENCY)
    economics = _json(ECONOMIC_REPLAY)
    extraction = _json(EXTRACT_DIR / "extraction_summary.json")
    checkpoint = _json(EXTRACT_DIR / "hour_checkpoint.json")
    coverage = pd.read_csv(COVERAGE_CSV)
    economic = pd.read_csv(ECONOMIC_SCENARIOS)
    model_comparison = pd.read_csv(MODEL_COMPARISON)
    trades = pd.read_parquet(TRADES)
    market_index = pd.read_parquet(EXTRACT_DIR / "market_index.parquet")

    model_path = ROOT / training["model_path"]
    candidate_model_path = ROOT / candidate["candidate_model_path"]
    candidate_calibrator_path = candidate_model_path.with_name("candidate_calibrator.json")
    training["model_file_sha256_verified"] = _sha256(model_path) if model_path.is_file() else None
    candidate["candidate_model_file_sha256_verified"] = _sha256(candidate_model_path) if candidate_model_path.is_file() else None
    candidate["candidate_model_file_exists"] = candidate_model_path.is_file()
    candidate["candidate_calibrator_path"] = candidate_calibrator_path.relative_to(ROOT).as_posix()
    candidate["candidate_calibrator_file_sha256_verified"] = _sha256(candidate_calibrator_path) if candidate_calibrator_path.is_file() else None
    candidate["candidate_calibrator_file_exists"] = candidate_calibrator_path.is_file()

    point_time = []
    for path in sorted(REPORT_DIR.glob("point_in_time_*.json")):
        point_time.append(_json(path))
    map_payload = _json(ROOT / "data/analysis/polymarket/BTC/preopen_v1/pmxt_v2_archive/market_token_map.json")
    pilot_rows = []
    ordered_markets = market_index.sort_values("market_start_utc", kind="stable").reset_index(drop=True)
    for label, position in (("beginning", 0), ("middle", len(ordered_markets) // 2), ("end", len(ordered_markets) - 1)):
        market = ordered_markets.iloc[position]
        coverage_row = coverage.loc[
            coverage.condition_id.eq(market.condition_id)
            & coverage.entry_case.eq("prestart_c45_o5")
        ].iloc[0]
        mapping = map_payload["markets"][market.condition_id]
        pilot_rows.append({
            "selection": label,
            "selection_rule": "chronological first, middle, and last official market; independent of predictions and outcomes",
            "condition_id": market.condition_id,
            "market_slug": market.market_slug,
            "market_start_utc": market.market_start_utc,
            "token_mapping_source": mapping.get("source"),
            "up_token_id": mapping.get("up_token_id"),
            "down_token_id": mapping.get("down_token_id"),
            "prestart_45s_compute_5s_order_coverage_reason": coverage_row.coverage_reason_age30s,
            "ask_bbo_mismatches_at_entry": int(coverage_row.bbo_at_entry_ask_mismatches),
            "quote_up_is_complemented": bool(coverage_row.up_quote_complemented),
            "quote_down_is_complemented": bool(coverage_row.down_quote_complemented),
            "market_event_count_by_entry": int(coverage_row.market_event_count),
            "initial_book_snapshots_by_entry": int(coverage_row.market_book_snapshot_count),
            "first_event_received_utc": coverage_row.market_first_event_utc,
            "last_event_received_utc": coverage_row.market_last_event_utc,
            "ask_up_usd": coverage_row.up_best_ask,
            "ask_down_usd": coverage_row.down_best_ask,
            "up_ask_age_seconds": coverage_row.up_ask_age_seconds,
            "down_ask_age_seconds": coverage_row.down_ask_age_seconds,
            "up_ask_received_age_seconds": coverage_row.up_ask_received_age_seconds,
            "down_ask_received_age_seconds": coverage_row.down_ask_received_age_seconds,
        })
    official_rows = evaluation["metrics"]["official_market_data"]["rows"]
    proxy_rows = evaluation["metrics"]["price_proxy"]["raw_model"]["n"]
    live_timing = {
        "historical_live_log_source": "docs/polymarket_btc_experiment.md, table 3; raw data/live/BTC/trade CSV and logs are not present in this checkout",
        "historical_live_raw_logs_available": False,
        "historical_live_run": "20260620_052109",
        "historical_live_model_matches_current": False,
        "historical_live_submission_attempts": 111,
        "historical_live_positive_filled_stake_response_rows": 89,
        "historical_live_filled_stake_missing_rows": 0,
        "historical_live_ack_timestamp_rows": 0,
        "historical_live_fill_timestamp_rows": 0,
        "historical_live_stages": [
            {"stage": "price_event_from_minute_open_ms", "n": 527, "p50_ms": 4.0, "p95_ms": 14.0, "p99_ms": 17.74, "anchor": "closed-candle close / following minute open; WS source event"},
            {"stage": "volume_event_from_minute_open_ms", "n": 527, "p50_ms": 191.0, "p95_ms": 865.4, "p99_ms": 1540.12, "anchor": "closed-candle close / following minute open; WS source event"},
            {"stage": "price_received_from_minute_open_ms", "n": 527, "p50_ms": 123.12, "p95_ms": 133.18, "p99_ms": 140.13, "anchor": "closed-candle close / following minute open; host wall clock"},
            {"stage": "volume_received_from_minute_open_ms", "n": 527, "p50_ms": 318.94, "p95_ms": 996.59, "p99_ms": 1667.34, "anchor": "closed-candle close / following minute open; host wall clock"},
            {"stage": "both_required_inputs_ready_from_minute_open_ms", "n": 527, "p50_ms": 321.38, "p95_ms": 996.59, "p99_ms": 1667.34, "anchor": "closed-candle close / following minute open; host wall clock"},
            {"stage": "feature_preparation_ms", "n": 527, "p50_ms": 0.43, "p95_ms": 0.66, "p99_ms": 1.01, "anchor": "local perf_counter"},
            {"stage": "feature_vector_construction_ms", "n": 527, "p50_ms": 31.02, "p95_ms": 45.66, "p99_ms": 49.53, "anchor": "local perf_counter"},
            {"stage": "model_inference_ms", "n": 527, "p50_ms": 1.36, "p95_ms": 1.75, "p99_ms": 2.51, "anchor": "local perf_counter"},
            {"stage": "signal_ready_from_window_start_ms", "n": 527, "p50_ms": 357.02, "p95_ms": 1033.15, "p99_ms": 1702.06, "anchor": "target window start, equal to closed-candle close in this live cycle; host wall clock"},
            {"stage": "quote_snapshot_lookup_or_refetch_ms", "n": 527, "p50_ms": 0.07, "p95_ms": 0.13, "p99_ms": 53.06, "anchor": "local perf_counter"},
            {"stage": "policy_decision_computation_ms", "n": 526, "p50_ms": 0.09, "p95_ms": 0.17, "p99_ms": 0.21, "anchor": "local perf_counter"},
            {"stage": "decision_ready_from_window_start_ms", "n": 526, "p50_ms": 357.5, "p95_ms": 1033.73, "p99_ms": 1780.79, "anchor": "target window start, equal to closed-candle close in this live cycle; host wall clock"},
            {"stage": "submit_call_including_response_ms", "n": 111, "p50_ms": 403.15, "p95_ms": 631.26, "p99_ms": 1001.86, "anchor": "local perf_counter; synchronous call and response, no separate send/ACK"},
            {"stage": "execution_stage_including_lookup_policy_and_submission_ms", "n": 527, "p50_ms": 0.21, "p95_ms": 448.6, "p99_ms": 818.19, "anchor": "local perf_counter; no-trade rows make median non-comparable to submitted calls"},
            {"stage": "cycle_complete_from_window_start_ms", "n": 527, "p50_ms": 475.28, "p95_ms": 1270.76, "p99_ms": 1791.68, "anchor": "target window start, equal to closed-candle close in this live cycle; host wall clock"},
        ],
        "preopen_collection": {
            "source": "docs/btc_preopen_experiment_manifest_20261004.json, collection-only run; orders disabled",
            "bundle_versions": ["btc-preopen-v1:4ed7e34d94bc42a0", "btc-preopen-v1:47c72bd8eb7e273a"],
            "decision_anchor": "market_start minus 60 seconds; candle opened at market_start minus 120 seconds",
            "feature_candle_local_receive": {"n": 1, "offset_after_decision_ms": {"p50": 1661.337, "p95": 1661.337, "p99": 1661.337}},
            "prediction_ready": {"n": 2, "offset_after_decision_ms": {"p50": 2174.508, "p95": 2614.4055, "p99": 2653.5075}, "observed_offsets_ms": [1685.733, 2663.283]},
            "all_execution_inputs_ready": {"n": 2, "offset_after_decision_ms": {"p50": 3106.706, "p95": 3605.3294, "p99": 3649.65148}, "observed_offsets_ms": [2552.68, 3660.732]},
            "prediction_to_all_inputs_ready": {"n": 2, "offset_ms": {"p50": 932.198, "p95": 990.7239, "p99": 996.92398}, "observed_offsets_ms": [866.947, 997.449]},
            "order_send_rows": 0,
            "exchange_ack_rows": 0,
            "fill_rows": 0,
            "clock_limit": "host UTC offsets were not calibrated; second observation records suspected source-book clock skew of -99.268 ms",
        },
        "historical_live_p50_test_reference": {
            "observation_count": 527,
            "signal_ready_from_candle_close_p50_ms": 357.02,
            "cycle_complete_from_candle_close_p50_ms": 475.28,
            "rounded_entry_delay_seconds": 1,
            "replay_case": "prestart_c0_o1",
            "entry_relative_to_market_start": "T-59s",
            "interpretation": "Usable historical p50 reference for timing tests, rounded upward to a one-second replay offset. This run used a different model/runtime and is not a current-candidate end-to-end measurement.",
        },
        "conservative_local_component_scenario": {
            "input_ready_observation_count": 2,
            "all_execution_inputs_ready_offsets_after_decision_seconds": [2.55268, 3.660732],
            "median_all_execution_inputs_ready_after_decision_seconds": 3.106706,
            "historical_local_submit_call_observation_count": 111,
            "historical_local_submit_call_p99_seconds": 1.00186,
            "median_component_sum_seconds_not_a_joint_quantile": 3.509856,
            "slowest_observed_input_plus_local_submit_p99_seconds_not_a_joint_quantile": 4.662592,
            "rounded_replay_delay_seconds": 5,
            "rounding_margin_seconds": 0.337408,
            "replay_case": "prestart_c0_o5",
            "entry_relative_to_market_start": "T-55s",
            "interpretation": "Conservative component-based local-host scenario. Input-readiness and submit-call measurements come from separate sessions and bundles, so their sum is not a measured end-to-end percentile. Submit-call duration includes client response; no separate exchange ACK or fill timestamp is available.",
        },
        "candidate_bundle_inference": inference_latency,
        "candidate_live_feature_path": {
            "candidate_feature_count": 112,
            "preopen_collection_v3_feature_count": 29,
            "preopen_collection_v3_model_meta": "data/models/BTC/btc_preopen_v1/lgbm_meta.json",
            "generic_current_live_feature_count": 256,
            "candidate_features_available_in_generic_current_live_model": 44,
            "candidate_incremental_feature_update_benchmark_available": False,
            "reason": "The 112-feature research candidate is not wired into the pre-open collector. That collector loads the 29-feature pre-open bundle. The generic live metadata has 256 columns, only 44 of which match the candidate; the candidate artifact directory contains a booster and calibrator, but no matching live feature updater/manifest. A full-history rebuild was not timed as live inference.",
        },
        "interpretation": [
            "The June historical log supports sub-second median receive-to-decision behavior in its different live model; the recorded p95/p99 are above one second.",
            "In the historical runtime, live_minute_opened is candle Opened plus one minute, so the WS 'minute open' and next target-window start coincide with the just-closed candle boundary. The wall-clock delay columns are close-anchored for that cycle, but have no recorded clock-offset calibration. Local feature/inference and submit-call durations are separate monotonic measurements and must not be summed as if their quantiles aligned.",
            "Synchronous submit duration includes the client response and is not a separately timestamped exchange acknowledgement or fill.",
            "A positive filled_stake_usdc response value is not an execution timestamp; the log summary reports 89 positive response rows among 111 attempts, with no fill timestamps.",
            "The two October pre-open collection observations measure prediction and quote-input availability only. Orders were disabled, so they do not measure submission, ACK, or execution.",
        ],
    }
    audit = {
        "verdict": "correct_with_limits",
        "verdict_scope": "The fitted v1 model has no confirmed training-target or feature-time leakage in the audited path. Its former raw replay verification used the wrong history cutoff, and model selection omitted a mandatory baseline comparison.",
        "confirmed_findings": [
            {
                "id": "original_replay_cutoff",
                "severity": "verification_bug",
                "detail": "The saved verification for Opened=2026-04-15T17:03Z rebuilt through 17:10Z instead of stopping at the 17:03 candle close. The corrected independent rebuild truncates raw inputs at the row's Opened time, rebuilds the 112 features, confirms target is absent, and exactly matches features and raw/Platt probabilities.",
                "corrected_anchor_count": len(point_time),
                "corrected_anchors": point_time,
                "prediction_changes_at_tested_anchors": 0,
            },
            {
                "id": "tuning_baseline_omission",
                "severity": "model_selection_limit",
                "detail": "Feature-selection score and model-tuning score use matching chronological folds, observation weights, and unweighted decision-row log loss. The tuning stage did not retain the feature-selector baseline as a competing trial or apply a baseline acceptance gate.",
                "selector_mean_logloss": 0.6919421088,
                "selector_std_logloss": 0.0008313739,
                "selector_mean_plus_half_std": 0.6923577958,
                "best_tuning_mean_plus_half_std": float(tuning["best_value"]),
                "fold_train_fingerprint": tuning["fold_indices"]["train_idx"],
                "fold_validation_fingerprint": tuning["fold_indices"]["valid_idx"],
                "best_trial_number": tuning["best_trial_number"],
            },
            {
                "id": "target_weight_objective",
                "severity": "methodological_choice",
                "detail": "Decision/auxiliary training weights were intentionally selected by balanced accuracy. This is not an implementation defect; it optimizes a classification metric rather than the probability log loss used for final model selection. The new candidate study selects weights with unweighted decision-row log loss.",
                "original_decision_weight": training["decision_weight"],
                "original_auxiliary_row_weight": training["auxiliary_row_weight"],
            },
            {
                "id": "external_period_exposure",
                "severity": "interpretation_limit",
                "detail": "The 2026-04-15 through 2026-05-18 period and overlapping BTC observations were used in earlier repository experiments. This audit reuses the period as authorized; it is historically exposed rather than a pristine holdout.",
            },
        ],
        "excluded_suspicions": [
            "The 66-round saved model is consistent with the best tuning optimum: trial 30's chosen objective is 0.692521820039, and the final model's parameters and 66-round schedule match that selected result. The 181-round log line is a fold-level early-stopping endpoint, not the final iteration-selection rule.",
            "No target-availability dependency was present in point-in-time prediction rebuilding; corrected checks confirmed target values were absent.",
            "No feature mismatch or prediction drift was found across the five corrected, hour/day-boundary replay anchors.",
        ],
        "original_model": {
            "run_path": str(RUN_DIR.relative_to(ROOT)).replace("\\", "/"),
            "run_id": run.get("task_identity"),
            "status": run.get("status"),
            "model_sha256": training["model_sha256"],
            "model_file_sha256_verified": training["model_file_sha256_verified"],
            "feature_count": len(training["feature_order"]),
            "feature_order_sha256": hashlib.sha256(json.dumps(training["feature_order"], separators=(",", ":")).encode()).hexdigest(),
            "generator_and_selection_stages": {
                "shared_fit_end_exclusive_utc": training["fit_end_exclusive"],
                "indicator_search": {
                    "config_count": indicator_stage["config_count"],
                    "target_mode": indicator_stage["target_mode"],
                    "label_available_before_fit_end": indicator_stage["label_available_before_fit_end"],
                },
                "volume_profile_search": {
                    "trials": volume_selection["trials_total"],
                    "best_trial": volume_selection["best_trial_number"],
                    "best_objective": volume_selection["best_value"],
                },
                "reaction_profile_search": {
                    "trials": reaction_selection["trials_total"],
                    "best_trial": reaction_selection["best_trial_number"],
                    "best_objective": reaction_selection["best_value"],
                },
                "feature_selection": {
                    "selected_feature_count": len(training["feature_order"]),
                    "objective_artifact": "chronological fold log loss; see the original feature-ranking and top-k sweep outputs",
                },
                "target_weight_selection": weight_selection,
            },
            "fit_start_utc": config["contract"]["chronological_split_frozen_before_tuning"]["feature_and_model_fit_start_utc"],
            "fit_end_exclusive_utc": training["fit_end_exclusive"],
            "latest_training_label_available_at_utc": training["latest_label_available_at"],
            "calibration": {
                "start_utc": config["contract"]["chronological_split_frozen_before_tuning"]["calibration_start_utc"],
                "end_exclusive_utc": config["contract"]["chronological_split_frozen_before_tuning"]["calibration_end_exclusive_utc"],
                "training_rows": run["stages"]["calibration"]["result"]["training_rows"],
                "latest_label_available_at_utc": evaluation["calibration_label_latest_available_at"],
                "source": "Binance COIN-M BTCUSD index proxy",
            },
            "training_rows": training["training_rows"],
            "best_iteration": training["best_iteration"],
            "params": training["params"],
            "target_formula": config["contract"]["training_target"]["formula"],
            "target_available_at": config["contract"]["training_target"]["label_available_at"],
            "tie_policy": config["contract"]["market"]["tie_policy"],
            "timezone": "UTC",
            "market_start_offset_seconds": config["contract"]["market"]["nominal_decision_offset_before_market_start_seconds"],
            "prediction_window_seconds": config["contract"]["market"]["prediction_window_seconds"],
            "fit_backend": config["compute_backend"],
            "threads": config["threads"],
            "folds": {
                "count": 10,
                "validation_decision_rows_reported": tuning["decision_rows"],
                "purge_rule": config["contract"]["chronological_split_frozen_before_tuning"]["internal_validation_label_purge"],
                "training_fingerprint": tuning["fold_indices"]["train_idx"],
                "validation_fingerprint": tuning["fold_indices"]["valid_idx"],
            },
            "best_trial": {
                "number": tuning["best_trial_number"],
                "objective_mean_plus_half_std": tuning["best_value"],
                "saved_iteration": tuning["best_iteration"],
                "successful_trials": tuning["successful_trials"],
                "total_trials": tuning["trials_total"],
                "validation_observation_weights": tuning["validation_weight"],
            },
            "cache_and_resume": {
                "run_status": run.get("status"),
                "resume_rule": config["contract"]["reproduction"]["resume"],
                "verified_policy": "stage reuse requires matching input/config/dependency identity and output hashes; no mismatched cached stage was accepted",
            },
        },
        "external_labels_and_metrics": {
            "external_market_start_first_utc": evaluation["external_test"]["market_start_first"],
            "external_market_start_last_utc": evaluation["external_test"]["market_start_last"],
            "official_polymarket_rows": official_rows,
            "binance_proxy_rows": proxy_rows,
            "proxy_official_disagreements": evaluation["metrics"]["official_market_data"]["price_proxy_disagreements"],
            "original_metrics": evaluation["metrics"],
            "old_paired_uncertainty_source": "Binance COIN-M BTCUSD proxy rows: the saved evaluation's paired_uncertainty is under metrics with label_source Binance and was compared on price_proxy labels. It is not an interval for official Polymarket outcomes.",
            "recomputed_paired_intervals": economics["predictive_intervals"],
            "recomputed_metrics_table": model_comparison.to_dict("records"),
        },
        "candidate_study": candidate,
        "live_timing": live_timing,
        "archive_and_replay": {
            "source_docs": [
                "https://archive.pmxt.dev/Polymarket/v2",
                "https://archive.pmxt.dev/docs/v2-data-overview",
                "https://docs.polymarket.com/market-data/realtime-data",
                "https://help.polymarket.com/en/articles/13364478-trading-fees",
                "https://help.polymarket.com/en/articles/14762452-polymarket-exchange-upgrade-april-28-2026",
                "https://github.com/Polymarket/ctf-exchange/blob/main/src/exchange/libraries/CalculatorHelper.sol",
                "https://github.com/Polymarket/ctf-exchange/docs/Overview.md",
                "https://github.com/Polymarket/ctf-exchange-v2/blob/main/src/exchange/mixins/Trading.sol",
            ],
            "observed_archive_schema": [
                "timestamp_received", "timestamp", "market", "event_type", "asset_id",
                "bids", "asks", "price", "size", "side", "best_bid", "best_ask",
                "fee_rate_bps", "transaction_hash", "old_tick_size", "new_tick_size",
            ],
            "event_ordering_limit": "PMXT rows expose source/receive timestamps and transaction_hash but no monotonic event sequence ID. Reconstruction sorts by receive time, source time, market, token, and event type; exact ties retain the order in the filtered Parquet part, but PMXT does not document that row order as an event sequence. Reported-BBO reconciliation is recorded, and ask-side disagreement at entry disqualifies the snapshot.",
            "market_token_pilot_samples": pilot_rows,
            "event_archive_identity": extraction.get("identity"),
            "hours_requested": extraction.get("hours_requested"),
            "hours_with_target_events": extraction.get("hours_with_target_events"),
            "hours_without_target_events": extraction.get("hours_without_target_events"),
            "selected_event_rows": extraction.get("selected_event_rows"),
            "hour_retry_attempts": extraction.get("hour_retry_attempts", 0),
            "hours_retried": extraction.get("hours_retried", []),
            "rowgroup_filter": extraction.get("row_group_filter"),
            "full_hourly_files_downloaded": extraction.get("full_hourly_files_downloaded"),
            "hour_checkpoint_count": len(checkpoint.get("hours", {})),
            "mapped_market_count": len(map_payload.get("markets", {})),
            "market_index_rows": int(pd.read_parquet(EXTRACT_DIR / "market_index.parquet", columns=["condition_id"]).shape[0]),
            "market_start_history_lookback_hours": extraction.get("identity", {}).get("history_lookback_hours"),
            "event_and_replay_summary": economics["event_archive"],
            "execution_assumptions": economics["money_and_execution"],
            "economic_scenarios": economic.to_dict("records"),
            "trade_log_rows": int(len(trades)),
            "coverage_rows": int(len(coverage)),
            "coverage_markets": int(coverage.condition_id.nunique()),
        },
        "run_stage_receipts": {
            name: {
                "status": stage.get("status"),
                "input_signature": stage.get("input_signature"),
                "artifact_signature": stage.get("artifact_signature"),
                "output_count": len(stage.get("outputs", [])),
            }
            for name, stage in run.get("stages", {}).items()
        },
    }
    (REPORT_DIR / "audit.json").write_text(json.dumps(_json_safe(audit), indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")

    main_case = "prestart_c0_o1"
    primary_cases = [main_case, "prestart_c0_o0", "prestart_c0_o2", "prestart_c0_o5"]
    fallback_case = "market_start_c45_o5"
    primary_scenarios = {
        case: _get_scenario(economic, case, 30, 60) for case in primary_cases
    }
    primary_coverages = {
        case: _coverage_counts(coverage, case, 30) for case in primary_cases
    }
    fallback_coverage_30 = _coverage_counts(coverage, fallback_case, 30)
    robustness_cases = ["prestart_c15_o5", "prestart_c45_o5"]
    robustness = {
        case: _get_scenario(economic, case, 30, 60) for case in robustness_cases
    }
    economics["primary_scenario"] = "prestart_c0_o1|age30|release60 (historical cycle-complete p50 rounded upward to a one-second entry delay; entry T-59s); ideal +0s, +2s sensitivity, and the conservative local-host T-55s scenario are also reported"
    economics["main_entry_times_relative_to_market_start"] = {
        "prestart_c0_o1": "T-59s (historical cycle-complete p50 of 475.28ms rounded upward to one second)",
        "prestart_c0_o0": "T-60s (ideal reference)",
        "prestart_c0_o2": "T-58s (+2s sensitivity)",
        "prestart_c0_o5": "T-55s (conservative local-host component scenario)",
    }
    economics["primary_latency_grid"] = {
        case: primary_scenarios[case] for case in primary_cases
    }
    economics["robustness_scenarios"] = {
        case: robustness[case] for case in robustness_cases
    }
    ECONOMIC_REPLAY.write_text(
        json.dumps(_json_safe(economics), indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    audit["archive_and_replay"]["primary_scenario"] = economics["primary_scenario"]
    audit["archive_and_replay"]["primary_latency_grid"] = economics["primary_latency_grid"]
    audit["archive_and_replay"]["robustness_scenarios"] = economics["robustness_scenarios"]
    (REPORT_DIR / "audit.json").write_text(
        json.dumps(_json_safe(audit), indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    candidate_dev = candidate["q3_2025_development_selection"]
    paired_official = candidate["paired_3day_bootstrap_vs_original"]["official_polymarket"]
    paired_proxy = candidate["paired_3day_bootstrap_vs_original"]["binance_proxy"]
    report_lines = [
        "# BTC Pre-Open v1: audit, economics, candidate study",
        "",
        "Raport końcowy z audytu uruchomienia `btc_preopen_v1`, odtworzenia punktu w czasie, treningu kandydata i replayu publicznych zdarzeń order booka. Nie wykonano rzeczywistych zleceń ani aktywacji live.",
        "",
        "## 1. Czy model wytrenowano poprawnie?",
        "",
        "**Werdykt: poprawny z ograniczeniami.** Nie znaleziono potwierdzonego wycieku targetu ani przyszłych danych w użytych cechach w zweryfikowanej ścieżce treningu. Potwierdzono błąd w dawnym teście replay: dla świecy `Opened=17:03` przyciął on historię dopiero do `17:10`. To błędne potwierdzenie punktu w czasie; sam zapisany model pozostał niezmieniony.",
        "",
        f"Nowy replay z surowego prefiksu sprawdził {len(point_time)} punktów, w tym granicę godziny i dnia. Każdy odtworzył **112/112 cech** oraz raw/Platt probability z maksymalną różnicą `0`; target nie był dostępny przy predykcji. Zasada czasu to świeca otwarta `T−2 min`, zamknięta `T−1 min`, decyzja `T−1 min`, rynek od `T` przez 5 minut; etykieta `Close[Opened+6m] >= Open[Opened+2m]`, remis UP, dostępna `Opened+7m`, UTC.",
        "",
        "- **A — 181 vs 66 iteracji:** `181` to early-stopping punkt pojedynczego foldu. Końcowy wybór skanuje iteracje na foldach i minimalizuje średni log loss + `0,5 × odchylenie`; trial 30 osiągnął `0.692521820039`, a finalny model zachował jego parametry i 66 iteracji. Rozbieżność nie wskazuje na zły checkpoint.",
        "- **B — 0.691942 vs 0.692147:** foldy, obserwacje i wagi są porównywalne; walidacja używa nieważonego log loss na minutach decyzyjnych. Selekcja cech dała `0.6919421` średniego LL i `0.6923578` po karze `0,5 × std`. Tuning wybrał `0.6925218`. Etap tuningu nie włączył selektora jako bazowego kandydata ani nie miał bramki akceptacji względem niego. To potwierdzona luka selekcji modelu.",
        "- **C — przyczynowość replayu:** pięć skorygowanych odtworzeń daje identyczne cechy i predykcje; istniejący test perturbacji przyszłego okna również pozostaje w zestawie testów. Nie stwierdzono wpływu świec przyszłych na wcześniejszą prognozę w tych sprawdzeniach.",
        "- **D — źródło etykiet:** historyczne `paired_uncertainty` w `evaluation.json` porównywało model z baseline na proxy Binance, nie na oficjalnym wyniku Polymarket. Poniżej i w `model_comparison.csv` log loss, Brier, AUC oraz paired 3-day block intervals są przeliczone osobno: proxy `n={proxy_rows}` i oficjalne Polymarket `n={official_rows}`. Etykiety rozeszły się w `400/{official_rows}` wspólnych rynków.",
        "- **E — wagi:** `decision_weight=0.23`, `auxiliary_row_weight=0.1925` dobrano celowo przez balanced accuracy. To wybór metodologiczny, nie błąd implementacji; nowy ograniczony search dobiera wagę przez nieważony log loss minut decyzyjnych.",
        "",
        f"Model oryginalny: SHA-256 `{training['model_sha256']}`, 112 cech, 66 iteracji, `{training['training_rows']:,}` wierszy treningowych, fit do `{training['fit_end_exclusive']}` (ostatnia dostępna etykieta `{training['latest_label_available_at']}`). Kalibracja kończy się przed pierwszą decyzją testową. Okres testowy był wcześniej analizowany w repozytorium, dlatego nie jest pristine holdoutem.",
        "",
        "## 2. Opóźnienia live i scenariusze czasu wejścia",
        "",
        "Surowe pliki `data/live/BTC/trade/*.csv` i `data/live/BTC/logs/*.log` nie są obecne w checkoutcie; poniższy rozkład historycznego runtime pochodzi z utrwalonej tabeli audytu w `docs/polymarket_btc_experiment.md` (run `20260620_052109`, model różny od obecnego). Są to `p50/p95/p99` i liczebności zapisane w tym raporcie, nie ponownie przeliczone próbki.",
        "",
        "| Etap z historycznego live | N | p50 | p95 | p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in live_timing["historical_live_stages"]:
        report_lines.append(
            f"| {stage['stage']} ({stage['anchor']}) | {stage['n']} | {stage['p50_ms']:.2f} ms | {stage['p95_ms']:.2f} ms | {stage['p99_ms']:.2f} ms |"
        )
    report_lines.extend([
        "",
        "W starym live kod uzywa `live_minute_opened = candle_opened + 1 min`, wiec kolumny opisane jako opoznienie od otwarcia minuty sa zakotwiczone w granicy zamkniecia wlasnie przetwarzanej swiecy. W tej sesji `p50` close-to-signal wynosil 357 ms, a `p50` close-to-cycle-end 475 ms, co wspiera obserwacje typowego cyklu ponizej sekundy. Ogon przekraczal sekunde: sygnal `p95=1.033 s`, cykl `p95=1.271 s`, `p99=1.792 s`. Czasy scienne hosta nie maja zapisanej kalibracji offsetu zegara. Koniec cyklu obejmuje wszystkie 527 decyzji, nie tylko 111 prob zlecenia; osobnego rozkladu close-to-send dla prob brak.",
        "Treat the historical p50 values as usable timing-test references: signal-ready p50 is 357.02ms and cycle-complete p50 is 475.28ms (N=527). The `prestart_c0_o1` case rounds the latter up to a one-second entry offset at T-59s; it is not a p95/p99 or a measurement of the current candidate.",
        "",
        "Z osobnego pre-open collection-only z 2026-10-04: decyzja nominalna była `T−60s`; pierwszy odbiór świecy zapisano `T−58.338663s`, gotową predykcję `T−58.314267s`, a wszystkie wejścia wykonawcze (w tym quote) `T−57.447320s`. To **jedna** obserwacja; druga ma predykcję gotową `T−57.336717s` i komplet wejść `T−56.339268s`. Dla dwóch próbek `p50` gotowości predykcji to 2.175 s po nominalnej decyzji, `p50` wszystkich wejść 3.107 s; `p95/p99` z `N=2` nie opisują wiarygodnie ogona. Zlecenia były wyłączone, więc liczebność send/ACK/fill wynosi zero.",
        "",
        f"Nowy kandydat LightGBM ma szybki, osobno zmierzony ciepły inference na 2,000 jednorzędowych wektorach 112 cech: p50 `{inference_latency['combined_model_predict_and_calibration']['p50_ms']:.3f} ms`, p95 `{inference_latency['combined_model_predict_and_calibration']['p95_ms']:.3f} ms`, p99 `{inference_latency['combined_model_predict_and_calibration']['p99_ms']:.3f} ms` razem z kalibracją. Pomiar nie obejmuje budowy cech. Pełny inkrementalny update tych 112 cech nie jest obecnie podłączony do pre-open collectora: jego bundle ma 29 cech, a ogólny live runtime ma 256 kolumn, z których 44 pokrywają się z kandydatem. Nie mierzono odbudowy historii jako inferencji live.",
        "",
        "Brak znaczników czasu wysłania, osobnego ACK giełdy i fillu. `submit_call_including_response_ms` (111 prób; `p50=403 ms`, `p95=631 ms`, `p99=1.002 s`) obejmuje synchroniczne wywołanie klienta z odpowiedzią; nie jest czasem fillu. Historyczny raport podaje dodatni `filled_stake_usdc` w 89 z 111 odpowiedzi, lecz nie zapisuje czasu ani niezależnego zdarzenia wykonania. W pre-open próbie nie było wysyłania zleceń.",
        "",
        "The timing-grounded main economic scenario uses the existing `prestart_c0_o1` snapshot: the historical cycle-complete p50 of 475.28ms is rounded upward to one second, so entry is T-59s after the nominal T-60s candle-close decision. The signal-ready p50 is 357.02ms (N=527). These are historical timing references, not measurements of the current candidate runtime.",
        "In the two pre-open collection observations, all execution inputs were ready +2.553s and +3.661s after the decision (N=2). The historical local submit-call p99 was 1.002s (N=111). The slower observed input time plus this local-call budget is 4.663s; rounding up to the existing five-second snapshot leaves 0.337s of margin.",
        "The T-55s row is a conservative local-host component scenario, not a measured joint p99: the input and submit-call measurements come from separate runs and bundles. Submit duration includes the client response, with no separate exchange ACK or fill time. The candidate incremental feature path still has no matching live updater.",
        "The T-60s row is an ideal reference; T-59s is the historical p50-based main scenario rounded upward from 475ms; T-58s is a +2s sensitivity. These old-live values are not current-candidate end-to-end measurements.",
        "",
        "| Scenario | Dokladny czas wejscia | Model | Saldo koncowe | PnL netto | Obrot | Fee | Drawdown | Transakcje | Brak danych | Bez przewagi |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for case in primary_cases:
        relative = {"prestart_c0_o1": "T-59s", "prestart_c0_o0": "T-60s", "prestart_c0_o2": "T-58s", "prestart_c0_o5": "T-55s"}[case]
        scenario_label = {"prestart_c0_o1": "Main, historical p50 ceiling (475ms to 1s)", "prestart_c0_o0": "Ideal reference", "prestart_c0_o2": "+2s sensitivity", "prestart_c0_o5": "Conservative local-host component"}[case]
        for row in primary_scenarios[case]:
            report_lines.append(
                f"| {scenario_label} ({case}) | {relative} | {row['model']} | {_money(row.get('ending_cash_usd'))} | {_money(row.get('net_pnl_usd'))} | {_money(row.get('gross_turnover_usd'))} | {_money(row.get('fees_paid_usd'))} | {float(row.get('max_drawdown_at_cost', 0.0)):.2%} | {int(row.get('trade_count', 0))} | {int(row.get('data_rejections', 0))} | {int(row.get('skip_no_positive_expected_edge', 0))} |"
            )
    report_lines.extend([
        "",
        "All rows use ask age <=30s, capital release 60s after resolution, initial cash $100 and gross $5 per position. T denotes market start. The main historical p50-based entry is T-59s; T-60s is ideal, T-58s is +2s sensitivity, and T-55s is a conservative local-host component scenario.",
        "",
        "Długie opóźnienia są osobnym testem odporności, nie głównym wynikiem: `prestart_c15_o5` wchodzi `T−40s`, `prestart_c45_o5` wchodzi `T−10s`. Nie należy z wyniku T−10s wnioskować o strategii wejścia T−60s.",
        "",
        "## 3. Wynik ekonomiczny",
        "",
        "Według schematu PMXT `timestamp_received` oznacza czas ingestu przez eksportera, a `timestamp` jest czasem źródłowym Polymarket; porównanie tych kolumn daje rozkład opóźnienia feedu.",
        f"Pobieranie odnotowało {extraction.get('hour_retry_attempts', 0)} ponownych prób dla godzin z błędem technicznym; szczegóły i wynik końcowy są zapisane w `audit.json`. Błędy HTTP nie są traktowane jako brak danych.",
        "Wykonano pełny, checkpointowany odczyt archiwum PMXT v2 dla 9 407 oficjalnych rynków. Każdy hourly Parquet filtrowano do docelowych condition ID i zdarzeń odebranych nie później niż `market_start + 5s`. BBO odbudowano z pełnego snapshotu i aktualizacji; zakup przechodzi po ask przez poziomy wystarczające na $5. Użyto `timestamp_received` jako czasu dostępności. Brak pełnego snapshotu, brak historycznego `fee_rate_bps`, niekompletny book, przeterminowany ask lub niewystarczająca głębokość powodują wykluczenie.",
        "",
        "Do replayu ekonomicznego wymagane s\u0105 natywne snapshoty token\u00f3w UP i DOWN oraz zgodno\u015b\u0107 ask z ostatnim raportowanym BBO; komplementarne kwotowania s\u0142u\u017c\u0105 wy\u0142\u0105cznie do diagnostyki i nie dostarczaj\u0105 g\u0142\u0119boko\u015bci do fillu.",
        "Przedzia\u0142y wieku ask liczono po czasie \u017ar\u00f3d\u0142owym ostatniej zmiany g\u0142\u0119boko\u015bci ask; `timestamp_received` ogranicza dost\u0119pno\u015b\u0107, a jego wiek jest zapisany osobno w CSV.",
        "PMXT nie zawiera monotonicznego identyfikatora kolejno\u015bci zdarze\u0144; przy identycznym czasie odbioru i \u017ar\u00f3d\u0142owym zachowano kolejno\u015b\u0107 w przefiltrowanym pliku Parquet, ale PMXT nie opisuje jej jako kolejno\u015bci zdarze\u0144. Zgodno\u015b\u0107 odtworzonego BBO jest raportowana, a niezgodno\u015b\u0107 ask przy wej\u015bciu wyklucza snapshot.",
        "Pilot wybrano chronologicznie dla pierwszego, środkowego i ostatniego rynku, niezależnie od predykcji i wyniku. Rzeczywiste odpowiedzi CLOB mapowały `condition_id` na tokeny UP/DOWN; próbki, pierwsze booki, eventy i ceny są w `audit.json`.",
        "",
        "Zastosowano regułę fee zgodną z datą wejścia. Przed modernizacją giełdy 28 kwietnia 2026 r. opłata BUY była potrącana w tokenach wyniku (`shares × rate × min(price, 1−price) / price`); replay zaokrągla zagregowane poziomy booka w dół do 6 miejsc, bo PMXT nie udostępnia wypełnień per maker. Dla wejść od 11:00 do 12:00 UTC przyjęto okno konserwatywnej przerwy i nie symulowano zleceń. Od 12:00 UTC opłata jest w collateral: `shares × rate × (price × (1−price))`, wykładnik 1 i aktualna precyzja 5 miejsc/minimum $0.00001; historyczny wykładnik nie występuje w archiwum. Book agreguje rozmiary zamiast pokazywać wypełnienia makerów, więc cash fee zaokrąglono dla całego fillu, a legacy fee per poziom; dokładne zaokrąglenie każdego matchu jest nieznane. `fee_rate_bps` pochodzi z ostatniego odebranego eventu przed wejściem. To przybliżenie nie odtwarza dokładnej minuty wznowienia ani rozliczenia opłaty dla poszczególnych makerów. Źródła: [opis modernizacji](https://help.polymarket.com/en/articles/14762452-polymarket-exchange-upgrade-april-28-2026), [stary wzór kontraktu](https://github.com/Polymarket/ctf-exchange/blob/main/src/exchange/libraries/CalculatorHelper.sol), [nowy settlement](https://github.com/Polymarket/ctf-exchange-v2/blob/main/src/exchange/mixins/Trading.sol). Czas compute, order delay i dostępność środków po resolution są scenariuszami, nie pomiarami.",
        "",
        f"Coverage dla wariantow glownych (ask age <= 30s): `{json.dumps(primary_coverages, ensure_ascii=False)}`. Kazdy wariant startuje od $100, pojedynczy gross zakup to $5, a kapital wraca 60 s po `resolved_at_utc`. Wyniki dotycza tylko pokrytych rynkow; brakujacych bookow i nieznanych fee nie imputowano.",
        "",
        "| Test odpornosci | Dokladny czas wejscia | Model | PnL netto | Obrot | Fee | Drawdown | Transakcje | Brak danych |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for case, relative in (("prestart_c15_o5", "T-40s"), ("prestart_c45_o5", "T-10s")):
        for row in robustness[case]:
            report_lines.append(
                f"| {case} | {relative} | {row['model']} | {_money(row.get('net_pnl_usd'))} | {_money(row.get('gross_turnover_usd'))} | {_money(row.get('fees_paid_usd'))} | {float(row.get('max_drawdown_at_cost', 0.0)):.2%} | {int(row.get('trade_count', 0))} | {int(row.get('data_rejections', 0))} |"
            )
    report_lines.extend([
        "",
        "Fallback jest oddzielny: `market_start_c45_o0` wchodzi w T+0s (predykcja gotowa T-15s); `market_start_c45_o5` wchodzi T+5s. Pelna macierz opoznien, wieku ask i czasu zwolnienia kapitalu pozostaje w `economic_scenarios.csv`.",
        "",
        "| Fallback | Dokladny czas wejscia | Model | PnL netto | Obrot | Fee | Drawdown | Transakcje | Brak danych |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for case, relative in (("market_start_c45_o0", "T+0s"), ("market_start_c45_o5", "T+5s")):
        for row in _get_scenario(economic, case, 30, 60):
            report_lines.append(
                f"| {case} | {relative} | {row['model']} | {_money(row.get('net_pnl_usd'))} | {_money(row.get('gross_turnover_usd'))} | {_money(row.get('fees_paid_usd'))} | {float(row.get('max_drawdown_at_cost', 0.0)):.2%} | {int(row.get('trade_count', 0))} | {int(row.get('data_rejections', 0))} |"
            )
    report_lines.extend([
        "",
        f"Coverage wariantu T+5s: `{json.dumps(fallback_coverage_30, ensure_ascii=False)}`. Drawdown uwzglednia gotowke plus koszt pozycji zablokowanych, bez niezrealizowanej zmiany wartosci w trakcie rynku. Symulowany ask fill nie jest dowodem rzeczywistego wykonania.",
        "",
        "Wcześniejszy `p_old_model_up` powstawał minutę później i miał dostęp do innej informacji, dlatego nie był porównywalny jako decyzja pre-open. Nie użyto polityki MARKET_ONLY.",
        "",
        "Archiwum i schemat: [PMXT Polymarket v2](https://archive.pmxt.dev/Polymarket/v2), [PMXT v2 data overview](https://archive.pmxt.dev/docs/v2-data-overview). Fee formula i modernizacja: [Polymarket Trading Fees](https://help.polymarket.com/en/articles/13364478-trading-fees), [Exchange Upgrade April 28](https://help.polymarket.com/en/articles/14762452-polymarket-exchange-upgrade-april-28-2026), [legacy CalculatorHelper](https://github.com/Polymarket/ctf-exchange/blob/main/src/exchange/libraries/CalculatorHelper.sol), [v2 Trading settlement](https://github.com/Polymarket/ctf-exchange-v2/blob/main/src/exchange/mixins/Trading.sol). Raport zapisuje godziny, hash części, liczbę row groups, mapowanie tokenów i opóźnienie receive/source.",
        "",
        "## 4. Wynik nowego treningu",
        "",
        f"Wytrenowano ograniczonego kandydata LightGBM GPU w 16 trialach, 4 wątkach i dwóch chronologicznych foldach. Kandydat i reguła wyboru zostały ustalone przed oceną Q3 2025; okres zewnętrzny nie służył wyborowi. Zwycięzca deweloperski: `tuned_search_winner`, historia 3 lata, waga decyzji `0.23`, 103 iteracje.",
        "",
        f"Na kwartale Q3 2025 log loss wyniósł `{candidate_dev['original_v1_recipe']['logloss']:.9f}` dla przepisu v1 i `{candidate_dev['tuned_search_winner']['logloss']:.9f}` dla kandydata. Sparowany 3-day bootstrap dla różnicy LL (kandydat − v1) ma 95% CI `[{candidate['development_paired_bootstrap_vs_original']['delta_first_minus_second_log_loss_ci95'][0]:.7f}, {candidate['development_paired_bootstrap_vs_original']['delta_first_minus_second_log_loss_ci95'][1]:.7f}]`, obejmujący zero. Na oficjalnych zewnętrznych etykietach przedział raw również obejmuje zero `[{paired_official['candidate_raw_minus_original_raw']['delta_first_minus_second_log_loss_ci95'][0]:.7f}, {paired_official['candidate_raw_minus_original_raw']['delta_first_minus_second_log_loss_ci95'][1]:.7f}]`. Proxy raw przedział to `[{paired_proxy['candidate_raw_minus_original_raw']['delta_first_minus_second_log_loss_ci95'][0]:.7f}, {paired_proxy['candidate_raw_minus_original_raw']['delta_first_minus_second_log_loss_ci95'][1]:.7f}]`.",
        "",
        "Kandydat nie wykazał stabilnej poprawy według ustalonej reguły. Zachowano model v1 jako wybraną konfigurację; model kandydata pozostaje porównaniem badawczym. Raw/Platt oraz oficjalne/Proxy metryki i sparowane przedziały są rozdzielone w `model_comparison.csv`.",
        "",
        "## Pliki dostawy",
        "",
        "- `audit.json` — wynik audytu i dane źródłowe do odtworzenia werdyktu.",
        "- `data_coverage.csv` — wszystkie 9 407 rynków dla 16 czasów wejścia.",
        "- `model_comparison.csv` — metryki raw/Platt dla obu etykiet oraz paired intervals.",
        "- `economic_scenarios.csv` i `economic_replay.json` — pełna macierz opóźnień, świeżości i zwalniania kapitału.",
        f"- `trades.parquet` — {len(trades):,} wierszy dziennika transakcji dla scenariuszy z ask freshness do 30 s.",
        "- `candidate_metrics.json`, `candidate_search_trials.csv` i pięć `point_in_time_*.json` — małe dowody treningu/replayu.",
        "- `candidate_inference_latency.json`: warm single-row candidate inference and calibration; feature generation is not included.",
        "- `report_bundle.zip` — raport, artefakty metadanych, wyniki i kod odtworzeniowy bez surowych shardów archiwum.",
        "",
        "Nie aktywowano modelu live ani nie wysłano zleceń.",
        "",
    ])
    report_lines = [_repair_report_line(line) for line in report_lines]
    (REPORT_DIR / "SUMMARY.md").write_text("\n".join(report_lines), encoding="utf-8")

    include_paths = [
        REPORT_DIR / "SUMMARY.md", REPORT_DIR / "audit.json", COVERAGE_CSV,
        MODEL_COMPARISON, ECONOMIC_SCENARIOS, ECONOMIC_REPLAY, TRADES,
        CANDIDATE_METRICS, REPORT_DIR / "candidate_search_trials.csv",
        INFERENCE_LATENCY, ROOT / "benchmark_btc_preopen_bundle.py",
        REPORT_DIR / "candidate_external_predictions.parquet",
        REPORT_DIR / "original_bundle_verification.json",
        ROOT / training["model_path"],
        ROOT / candidate["candidate_model_path"],
        candidate_calibrator_path,
        ROOT / "data/analysis/polymarket/BTC/preopen_v1/candidate_study_20261005/study_identity.json",
        ROOT / "data/analysis/polymarket/BTC/preopen_v1/candidate_study_20261005/frozen_candidate_list.json",
        ROOT / "data/analysis/polymarket/BTC/preopen_v1/candidate_study_20261005/selection_results.json",
        EXTRACT_DIR / "hour_checkpoint.json",
        *sorted(REPORT_DIR.glob("point_in_time_*.json")),
        ORIGINAL_TRAINING_MANIFEST, ORIGINAL_EVALUATION, ORIGINAL_MODEL_TUNING,
        ORIGINAL_CONFIG, EXTRACT_DIR / "extraction_summary.json", EXTRACT_DIR / "archive_identity.json",
        ROOT / "docs/polymarket_btc_experiment.md",
        ROOT / "docs/btc_preopen_experiment_manifest_20261004.json",
        ROOT / "run.py", ROOT / "audit_btc_oof.py",
        ORIGINAL_RUN_MANIFEST,
        RUN_DIR / "stages/calibration_6942659d3b63/platt_calibrator.json",
        RUN_DIR / "stages/feature_selection_2eeac6e4f194/selected_features.json",
        RUN_DIR / "stages/feature_selection_2eeac6e4f194/feature_ranking.parquet",
        RUN_DIR / "stages/feature_selection_2eeac6e4f194/topk_sweep.parquet",
        RUN_DIR / "stages/final_model_51d3adab0c53/feature_order.json",
        RUN_DIR / "stages/final_model_51d3adab0c53/feature_sources.json",
        RUN_DIR / "stages/final_model_51d3adab0c53/lgbm_meta.json",
        RUN_DIR / "stages/volume_profile_d9bd4bb26193/best_result.json",
        RUN_DIR / "stages/volume_profile_d9bd4bb26193/fitted_generator_config.json",
        RUN_DIR / "stages/reaction_profile_47fc0f8c0603/best_result.json",
        RUN_DIR / "stages/reaction_profile_47fc0f8c0603/fitted_generator_config.json",
        RUN_DIR / "stages/target_weights_7f8808f29356/target_weight_result.json",
        RUN_DIR / "stages/target_weights_7f8808f29356/candidate_fold_metrics.parquet",
        RUN_DIR / "stages/indicators_b150948edf4d/indicator_stage.json",
        ROOT / "run_btc_preopen_experiment.py", ROOT / "run_btc_preopen_candidate_study.py",
        ROOT / "run_btc_preopen_pmxt_extract.py", ROOT / "run_btc_preopen_economic_replay.py",
        ROOT / "build_btc_preopen_report.py",
        ROOT / "features/btc_preopen_contract.py", ROOT / "tests/test_btc_preopen_contract.py",
        ROOT / "tests/test_btc_preopen_candidate_study.py", ROOT / "tests/test_btc_preopen_economic_replay.py",
        ROOT / "tests/test_btc_preopen_pmxt_extract.py",
    ]
    bundle_path = REPORT_DIR / "report_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in include_paths:
            if path.is_file():
                bundle.write(path, path.relative_to(ROOT).as_posix())
    print(json.dumps({
        "summary": str((REPORT_DIR / "SUMMARY.md").relative_to(ROOT)),
        "audit": str((REPORT_DIR / "audit.json").relative_to(ROOT)),
        "coverage_rows": len(coverage),
        "model_comparison_rows": len(model_comparison),
        "economic_scenario_rows": len(economic),
        "trade_rows": len(trades),
        "bundle_bytes": bundle_path.stat().st_size,
    }, indent=2), flush=True)


if __name__ == "__main__":
    build()
