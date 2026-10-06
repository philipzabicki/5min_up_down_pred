"""Audit BTC pre-open book coverage, candidate feature wiring, and live parity."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
os.environ["POLYMARKET_RUNTIME_CONFIG_PATH"] = str(
    ROOT / "configs/runtime/btc_preopen_candidate.json"
)

import audit_feature_readiness as audit  # noqa: E402
import run as live_runtime  # noqa: E402
import run_btc_preopen_economic_replay as replay  # noqa: E402


REPORT_DIR = ROOT / "reports/btc_preopen"
RUN_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/nightly_v1/06dc69ccb35a63b3"
MODEL_STAGE_DIR = RUN_DIR / "stages/final_model_51d3adab0c53"
DATASET_DIR = RUN_DIR / "stages/final_feature_dataset_1d21891c3f3f/dataset"
PROFILE_STATE_DIR = RUN_DIR / "stages/final_feature_dataset_1d21891c3f3f/states"
ARCHIVE_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/pmxt_v2_archive_full_scenarios"
PARTS_DIR = ARCHIVE_DIR / "event_parts"
RAW_PATH = ROOT / "data/datasets/raw/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m.csv"
FEATURE_PARQUET = DATASET_DIR / "BTCUSD_INDEXVOL_UM_BTCUSDT1m_preopen_v1.parquet"
CANDIDATE_META = ROOT / "configs/runtime/btc_preopen_candidate_model_meta.json"
CANDIDATE_FEATURE_CONFIG = ROOT / "configs/runtime/btc_preopen_candidate_features.json"
CANDIDATE_HISTORY = ROOT / "configs/runtime/btc_preopen_candidate_history_requirements.json"
CANDIDATE_RUNTIME = ROOT / "configs/runtime/btc_preopen_candidate.json"
INDICATOR_FIT_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/nightly_v1/06dc69ccb35a63b3/stages/indicators_b150948edf4d/results/ba1c9334ea457365"
UP_ENTRY_CASE = "prestart_c0_o0"
MAIN_ENTRY_CASE = "prestart_c0_o1"
ASK_AGE_SECONDS = 30
PREDICTION_DELAY_SECONDS = 0
ENTRY_ORDER_DELAY_SECONDS = 1
RELEASE_DELAY_SECONDS = 60
FEATURE_ABS_TOL = 1e-6
FEATURE_REL_TOL = 1e-5
PROBABILITY_ABS_TOL = 1e-6
PARITY_START = pd.Timestamp("2026-04-15T17:03:00Z")
PARITY_END = pd.Timestamp("2026-04-16T00:04:00Z")
RESUME_AT = pd.Timestamp("2026-04-15T23:58:00Z")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_order(columns):
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


def _git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_last_commit(path):
    relative_path = Path(path).relative_to(ROOT).as_posix()
    try:
        output = subprocess.check_output(
            ["git", "log", "-1", "--format=%H%n%s", "--", relative_path],
            cwd=ROOT,
            text=True,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None
    if not output:
        return None
    return {"commit": output[0], "subject": output[1] if len(output) > 1 else ""}


def _as_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _target_replay_identity(target_coverage):
    entries = target_coverage.copy()
    entries["entry_time_utc"] = pd.to_datetime(
        entries.entry_time_utc, utc=True, format="mixed"
    )
    entries["market_first_event_utc"] = pd.to_datetime(
        entries.market_first_event_utc, utc=True, format="mixed"
    )
    target_fingerprint = hashlib.sha256(
        "\n".join(
            f"{row.condition_id}:{row.entry_case}:{int(row.entry_time_utc.value)}"
            for row in entries.sort_values(
                ["condition_id", "entry_case"], kind="stable"
            ).itertuples(index=False)
        ).encode("utf-8")
    ).hexdigest()
    first_hour = entries.market_first_event_utc.min().floor("h")
    last_hour = entries.entry_time_utc.max().floor("h")
    part_paths = []
    for path in sorted(PARTS_DIR.glob("*.parquet")):
        partition_hour = pd.Timestamp(path.stem.replace("T", " "), tz="UTC")
        if first_hour <= partition_hour <= last_hour:
            part_paths.append(path)
    part_signature = [
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in part_paths
    ]
    archive_fingerprint = hashlib.sha256(
        "\n".join(
            f"{name}:{size}:{mtime}" for name, size, mtime in part_signature
        ).encode("utf-8")
    ).hexdigest()
    archive_fingerprint = hashlib.sha256(
        "\n".join(f"{name}:{size}:{mtime}" for name, size, mtime in part_signature)
        .encode("utf-8")
    ).hexdigest()
    return target_fingerprint, archive_fingerprint, len(part_paths)


def _date_key(values):
    return pd.to_datetime(values, utc=True, errors="coerce").dt.strftime("%Y-%m-%d")


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _classification_map(feature_columns):
    parts = live_runtime.split_feature_subset(
        feature_columns,
        source_label=str(CANDIDATE_META),
    )
    family_columns = {
        "indicator": {spec.feature_col for spec in live_runtime.load_indicator_specs(
            feature_columns,
            source_label=str(CANDIDATE_META),
            fit_results_dir=INDICATOR_FIT_DIR,
        )},
        "volume_profile": set(parts["volume_profile_feature_cols"]),
        "reaction_profile": set(parts["reaction_profile_feature_cols"]),
        "basis_premium": set(parts["basis_premium_feature_cols"]),
        "realized_volatility": set(parts["realized_volatility_feature_cols"]),
        "session": set(parts["session_feature_cols"]),
        "candle": set(parts["candle_feature_cols"]),
        "streak": set(parts["streak_feature_cols"]),
    }
    result = {}
    for family, columns in family_columns.items():
        for column in columns:
            if column in result:
                raise RuntimeError(f"Feature classified into multiple families: {column}")
            result[column] = family
    missing = [column for column in feature_columns if column not in result]
    if missing:
        raise RuntimeError(f"Candidate features are not classified: {missing[:10]}")
    return result


def _feature_parameters(column, family, indicator_by_column, feature_config, history):
    if family == "indicator":
        config = indicator_by_column[column]
        requirement = history.get("runtime_window_by_feature", {}).get(
            column,
            history.get("required_stable_window_by_feature", {}).get(column),
        )
        return {
            "indicator": config["indicator"],
            "horizon": config["horizon"],
            "target_mode": config["target_mode"],
            "population": config["pop_size"],
            "fitted_params": config["params"],
            "required_candles_parameter_warmup": requirement,
        }
    if family == "volume_profile":
        config = feature_config["volume_profile_fixed_range"]
        return {
            "output": column,
            "horizons": config.get("horizons"),
            "neighbor_bins": config.get("neighbor_bins"),
            "config_signature_sha256": hashlib.sha256(
                config.get("config_signature", "").encode("utf-8")
            ).hexdigest(),
        }
    if family == "reaction_profile":
        config = feature_config["reaction_profile_fixed_grid"]
        return {
            "output": column,
            "horizons": config.get("horizons"),
            "neighbor_bins": config.get("neighbor_bins"),
            "config_signature_sha256": hashlib.sha256(
                config.get("config_signature", "").encode("utf-8")
            ).hexdigest(),
        }
    if family == "basis_premium":
        match = re.search(r"_(1m|3m|5m|15m|30m|1h|4h|1d)$", column)
        return {
            "interval": None if match is None else match.group(1),
            "statistic": column.rsplit("_", 1)[0],
            "config": feature_config["basis_premium_features"],
        }
    if family == "candle":
        interval_match = re.search(r"_(1m|3m|5m|15m|30m|1h|4h|1d)(?:_lag(\d+))?$", column)
        return {
            "definition": column.split("_", 1)[1],
            "interval": None if interval_match is None else interval_match.group(1),
            "lag": None if interval_match is None or interval_match.group(2) is None else int(interval_match.group(2)),
        }
    if family == "streak":
        return {"interval": column.removeprefix("candle_streak_")}
    if family == "session":
        return {"session_feature": column}
    if family == "realized_volatility":
        return {"window_or_contrast": column.removeprefix("realized_volatility_")}
    return {"feature": column}


def _feature_functions(family, column):
    if family == "indicator":
        indicator = column.split("_fit_", 1)[0]
        training = "create_modeling_dataset.add_indicator_values -> features.%s.get_%s_values" % (
            indicator,
            {
                "BollingerBands": "bollinger_bands",
                "ChaikinOsc": "chaikin_oscillator",
                "KeltnerChannel": "keltner_channel",
                "StochOsc": "stochastic_oscillator",
            }[indicator],
        )
        if indicator == "ChaikinOsc":
            return training, (
                "run.LivePredictor._initialize_indicator_state -> "
                "ChaikinOscillatorRuntimeState; _append_new_candle"
            )
        return training, "run.LivePredictor.load_indicator_specs -> features.live_indicator_runtime.LATEST_VALUE_BUILDERS"
    functions = {
        "candle": (
            "features.candle_features.build_candle_features / build_candle_feature_matrix",
            "run.LivePredictor._build_feature_vector -> build_latest_candle_derived_feature_dict_fast / build_latest_candle_pattern_feature_dict_fast / build_latest_candle_streak_feature_dict_fast",
        ),
        "streak": (
            "features.candle_features.add_candle_streak_features",
            "run.LivePredictor._build_feature_vector -> build_latest_candle_streak_feature_dict_fast",
        ),
        "session": (
            "features.session_open_features.add_session_open_features",
            "run.LivePredictor._build_feature_vector -> build_latest_session_open_feature_dict_fast",
        ),
        "realized_volatility": (
            "features.realized_volatility.add_realized_volatility_features",
            "run.LivePredictor._initialize_realized_volatility_state -> RealizedVolatilityRuntimeState.update",
        ),
        "basis_premium": (
            "features.basis_premium_features.add_basis_premium_features",
            "run.LivePredictor._initialize_basis_premium_state / _build_latest_basis_premium_features",
        ),
        "volume_profile": (
            "features.volume_profile_fixed_range.build_volume_profile_feature_matrix_from_arrays",
            "run.LivePredictor._prepare_volume_profile_features_for_latest_candle -> update_state_with_candle / extract_features_from_state",
        ),
        "reaction_profile": (
            "features.reaction_profile_fixed_grid.build_reaction_profile_feature_matrix_from_arrays",
            "run.LivePredictor._prepare_reaction_profile_features_for_latest_candle -> update_state_with_candle / extract_features_from_state",
        ),
    }
    return functions[family]


def build_feature_map():
    meta = _json(CANDIDATE_META)
    feature_columns = list(meta["feature_columns"])
    feature_config = _json(CANDIDATE_FEATURE_CONFIG)
    history = _json(CANDIDATE_HISTORY)
    source_manifest = _json(MODEL_STAGE_DIR / "feature_sources.json")
    active_runtime = _json(ROOT / "configs/runtime/active.json")
    active_meta_path = ROOT / active_runtime["assets"]["BTC"]["artifacts"]["model_meta_path"]
    active_feature_columns = list(_json(active_meta_path)["feature_columns"])
    active_position = {name: index for index, name in enumerate(active_feature_columns, start=1)}
    collector_meta_path = ROOT / "data/models/BTC/btc_preopen_v1/lgbm_meta.json"
    collector_feature_columns = list(_json(collector_meta_path)["feature_columns"])
    source_order = source_manifest["feature_order"]
    if source_order != feature_columns:
        raise RuntimeError("Candidate feature metadata differs from training feature_sources order")

    family_by_column = _classification_map(feature_columns)
    fit_configs = live_runtime.parse_fit_results(INDICATOR_FIT_DIR)
    indicator_by_column = {
        item["feature_col"]: item for item in fit_configs
        if item["feature_col"] in family_by_column
    }
    rows = []
    for position, column in enumerate(feature_columns, start=1):
        family = family_by_column[column]
        training_function, live_function = _feature_functions(family, column)
        if family == "indicator":
            source = "Closed BTC index OHLCV candles; candidate-specific fit JSON in indicator_fit_results_dir"
            history_description = (
                f"{history.get('global_required_stable_window')} required stable candles; "
                f"parameter-based warm-up={history.get('required_stable_window_by_feature', {}).get(column)}; "
                "live indicator builders use the complete retained OHLCV history"
            )
            artifact = f"{INDICATOR_FIT_DIR.relative_to(ROOT).as_posix()}/<fit result containing {column}>"
            status = "implemented_and_candidate_configured"
            evidence = "load_indicator_specs resolves exact feature column to candidate fit JSON and live builder"
            fitted_params = indicator_by_column[column]["params"]
            if (
                    indicator_by_column[column]["indicator"] == "ChaikinOsc"
                    and fitted_params.get("fast_ma_type") == "EMA"
                    and fitted_params.get("slow_ma_type") == "SHMMA"
            ):
                source = "Closed BTC index OHLCV from the fitted feature history start"
                history_description = (
                    "Cumulative ADL, EMA, and exact SHMMA recurrence are restored from "
                    "the candidate seed and advanced on each closed candle; other "
                    "indicators use the full retained 21,600-candle history"
                )
                artifact = (
                    "configs/runtime/btc_preopen_candidate.json -> indicator_state_seed_path; "
                    "configs/runtime/btc_preopen_candidate_indicator_state.json"
                )
                status = "implemented_and_seeded_for_candidate"
                evidence = (
                    "seed is tied to model SHA and fitted periods; audit restores it, "
                    "checks chronological updates/resumption, and matches the training vector"
                )
        elif family == "volume_profile":
            source = "Closed BTC index OHLCV High/Low/Volume"
            history_description = "Causal state restored from the candidate profile artifact or initialized from the complete available prefix"
            artifact = CANDIDATE_RUNTIME.relative_to(ROOT).as_posix() + " -> volume_profile_modeling_state_path"
            status = "implemented_and_candidate_configured"
            evidence = "candidate config signature validates against model metadata; parity audit rebuilds state only from candles at/before each anchor"
        elif family == "reaction_profile":
            source = "Closed BTC index OHLCV Open/High/Low/Close"
            history_description = "Causal state restored from the candidate profile artifact or initialized from the complete available prefix"
            artifact = CANDIDATE_RUNTIME.relative_to(ROOT).as_posix() + " -> reaction_profile_modeling_state_path"
            status = "implemented_and_candidate_configured"
            evidence = "candidate config signature validates against model metadata; parity audit rebuilds state only from candles at/before each anchor"
        elif family == "basis_premium":
            source = "BTC index Close and Binance UM_BTCUSDT_Close"
            history_description = "Closed one-minute index/futures bars aggregated causally to the feature interval"
            artifact = CANDIDATE_FEATURE_CONFIG.relative_to(ROOT).as_posix()
            status = "implemented_and_candidate_configured"
            evidence = "startup requires the configured futures close column; live feature vector fails if the feature is not produced"
        elif family == "realized_volatility":
            source = "Closed BTC index Close values"
            history_description = "Rolling return state rebuilt from retained closed candles; global runtime window is recorded below"
            artifact = "No fitted artifact; fixed feature definition in features/realized_volatility.py"
            status = "implemented"
            evidence = "feature columns classify into RealizedVolatilityRuntimeState and produce values in live vector"
        elif family == "session":
            source = "UTC candle timestamp plus configured session calendar"
            history_description = "No price warm-up; time-zone calendar definitions are source code"
            artifact = "features/session_open_features.py session definitions"
            status = "implemented"
            evidence = "feature columns classify into supported session-open builders"
        elif family == "streak":
            source = "Closed BTC index OHLCV candles aggregated to the feature interval"
            history_description = "Interval-specific consecutive candle direction; uses completed candles and the retained 21,600-candle runtime buffer"
            artifact = "features/candle_features.py streak feature catalog"
            status = "implemented"
            evidence = "selected interval resolves through resolve_streak_interval_to_rule and the live latest-value builder"
        else:
            source = "Closed BTC index OHLCV candles"
            history_description = "Largest selected interval is 1d with lags from the feature name; runtime buffer is 21,600 candles"
            artifact = "features/candle_features.py feature catalog"
            status = "implemented"
            evidence = "all selected candle names resolve through the live candle feature catalog"
        rows.append({
            "position_1_based": position,
            "feature": column,
            "in_active_general_runtime_model": column in active_position,
            "active_general_runtime_position_1_based": active_position.get(column),
            "in_29_feature_preopen_baseline": column in set(collector_feature_columns),
            "family": family,
            "parameters": json.dumps(
                _feature_parameters(column, family, indicator_by_column, feature_config, history),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "training_function": training_function,
            "live_function": live_function,
            "required_data_sources": source,
            "history_and_warmup": history_description,
            "configuration_or_artifact": artifact,
            "runtime_status": status,
            "evidence": evidence,
        })
    output = REPORT_DIR / "feature_compatibility_112.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return rows


def _strict_reason_counts(coverage, entry_case):
    frame = coverage.loc[coverage.entry_case.eq(entry_case)].copy()
    frame["quote_valid"] = frame["quote_valid_strict_bid_lt_ask"].astype(bool)
    return frame.apply(
        lambda row: replay._data_reason(row, ASK_AGE_SECONDS),
        axis=1,
    ).value_counts().astype(int).to_dict()


def _replay_fresh_bbo_cases(
        target_coverage,
        *,
        checkpoint_enabled,
        write_outputs=True,
):
    event_columns = [
        "timestamp_received", "timestamp", "market", "event_type", "asset_id",
        "bids", "asks", "price", "size", "side", "best_bid", "best_ask",
        "fee_rate_bps",
    ]
    target_coverage = target_coverage.copy()
    target_coverage["entry_time_utc"] = pd.to_datetime(
        target_coverage.entry_time_utc,
        utc=True,
        format="mixed",
    )
    target_coverage["market_first_event_utc"] = pd.to_datetime(
        target_coverage.market_first_event_utc,
        utc=True,
        format="mixed",
    )
    condition_ids = sorted(target_coverage.condition_id.astype(str).unique())
    condition_id_set = set(condition_ids)
    market_index = pd.read_parquet(
        ARCHIVE_DIR / "market_index.parquet",
        columns=[
            "condition_id", "up_token_id", "down_token_id", "market_start_utc",
            "resolved_at_utc", "target_polymarket_up", "market_slug",
        ],
    )
    market_index["condition_id"] = market_index.condition_id.map(_as_text)
    market_index = market_index.loc[
        market_index.condition_id.isin(condition_id_set)
    ].copy()
    if len(market_index) != len(condition_ids):
        raise RuntimeError("Target BBO freshness replay is missing token mappings")

    cached_columns = [
        "condition_id", "entry_case", "p_model_raw", "p_model_platt",
        "p_candidate_raw", "p_candidate_platt", "market_event_count",
        "market_book_snapshot_count", "book_snapshot_token_count", "has_full_snapshot",
        "up_best_bid", "up_best_ask", "up_best_ask_size_shares",
        "up_ask_age_seconds", "up_ask_received_age_seconds",
        "down_best_bid", "down_best_ask", "down_best_ask_size_shares",
        "down_ask_age_seconds", "down_ask_received_age_seconds", "fee_known", "fee_rate_bps",
    ]
    cached = pd.read_parquet(REPORT_DIR / "entry_snapshots.parquet", columns=cached_columns)
    cached["condition_id"] = cached.condition_id.map(_as_text)
    cached = cached.loc[
        cached.condition_id.isin(condition_id_set)
        & cached.entry_case.isin([UP_ENTRY_CASE, MAIN_ENTRY_CASE])
    ].copy()
    market_values = cached.drop_duplicates("condition_id").set_index("condition_id")
    market_by_id = {}
    for market in market_index.to_dict("records"):
        condition_id = str(market["condition_id"])
        base = market_values.loc[condition_id]
        market_by_id[condition_id] = {
            **market,
            "p_model_raw": base.p_model_raw,
            "p_model_platt": base.p_model_platt,
            "p_candidate_raw": base.p_candidate_raw,
            "p_candidate_platt": base.p_candidate_platt,
        }

    cases_by_market = {}
    schedule = {}
    for row in target_coverage.itertuples(index=False):
        condition_id = str(row.condition_id)
        entry_time = pd.Timestamp(row.entry_time_utc)
        case = {
            "case_id": str(row.entry_case),
            "kind": str(row.entry_kind),
            "compute_delay_seconds": int(row.compute_delay_seconds),
            "order_delay_seconds": int(row.order_delay_seconds),
            "prediction_available_at": pd.Timestamp(row.prediction_available_at_utc),
            "entry_time": entry_time,
        }
        schedule.setdefault(int(entry_time.value), []).append((condition_id, case))
        cases_by_market.setdefault(condition_id, []).append(case)
    schedule_times = sorted(schedule)
    deadlines = {
        condition_id: max(int(case["entry_time"].value) for case in cases)
        for condition_id, cases in cases_by_market.items()
    }
    first_hour = target_coverage.market_first_event_utc.min().floor("h")
    last_hour = target_coverage.entry_time_utc.max().floor("h")
    part_paths = []
    for path in sorted(PARTS_DIR.glob("*.parquet")):
        partition_hour = pd.Timestamp(path.stem.replace("T", " "), tz="UTC")
        if first_hour <= partition_hour <= last_hour:
            part_paths.append(path)
    part_signature = [
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in part_paths
    ]
    target_fingerprint = hashlib.sha256(
        "\n".join(
            f"{row.condition_id}:{row.entry_case}:{int(row.entry_time_utc.value)}"
            for row in target_coverage.sort_values(
                ["condition_id", "entry_case"], kind="stable"
            ).itertuples(index=False)
        ).encode("utf-8")
    ).hexdigest()

    checkpoint_path = REPORT_DIR / "quote_freshness_replay_checkpoint_v1.pkl"
    states = {}
    replayed = {}
    start_position = 0
    schedule_position = 0
    if checkpoint_enabled and checkpoint_path.is_file():
        with checkpoint_path.open("rb") as stream:
            checkpoint = pickle.load(stream)
        if (
            checkpoint.get("target_fingerprint") != target_fingerprint
            or checkpoint.get("part_signature") != part_signature
            or checkpoint.get("replay_semantics_version") != 3
        ):
            raise RuntimeError(
                "Existing targeted BBO checkpoint does not match the current input identity"
            )
        states = checkpoint["states"]
        replayed = checkpoint["snapshots"]
        start_position = int(checkpoint["last_part_position"]) + 1
        schedule_position = int(checkpoint["schedule_position"])
        print(
            f"[bbo-refresh] resumed after {checkpoint['last_part_name']}: "
            f"snapshots={len(replayed):,}",
            flush=True,
        )

    def save_checkpoint(part_position, part_name):
        payload = {
            "replay_semantics_version": 3,
            "target_fingerprint": target_fingerprint,
            "archive_partition_fingerprint": archive_fingerprint,
            "part_signature": part_signature,
            "last_part_position": part_position,
            "last_part_name": part_name,
            "schedule_position": schedule_position,
            "states": states,
            "snapshots": replayed,
        }
        temporary_path = checkpoint_path.with_suffix(".pkl.tmp")
        with temporary_path.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, checkpoint_path)

    def capture_before(cutoff_ns, inclusive=False):
        nonlocal schedule_position
        while schedule_position < len(schedule_times):
            at_ns = schedule_times[schedule_position]
            if at_ns > cutoff_ns or (at_ns == cutoff_ns and not inclusive):
                break
            for condition_id, case in schedule[at_ns]:
                market = market_by_id[condition_id]
                state = states.get(condition_id, replay._new_state())
                snapshot = replay._snapshot(state, market, case)
                replayed[(condition_id, case["case_id"])] = {
                    "condition_id": condition_id,
                    "entry_case": case["case_id"],
                    "bbo_at_entry_checks": snapshot["bbo_at_entry_checks"],
                    "bbo_at_entry_mismatches": snapshot["bbo_at_entry_mismatches"],
                    "bbo_at_entry_ask_checks": snapshot["bbo_at_entry_ask_checks"],
                    "bbo_at_entry_ask_mismatches": snapshot["bbo_at_entry_ask_mismatches"],
                    "bbo_at_entry_not_newer_than_side_state": snapshot[
                        "bbo_at_entry_not_newer_than_side_state"
                    ],
                    "bbo_at_entry_older_than_side_state": snapshot[
                        "bbo_at_entry_older_than_side_state"
                    ],
                    "bbo_at_entry_tied_with_side_state": snapshot[
                        "bbo_at_entry_tied_with_side_state"
                    ],
                    "market_event_count": snapshot["market_event_count"],
                    "market_book_snapshot_count": snapshot["market_book_snapshot_count"],
                    "book_snapshot_token_count": snapshot["book_snapshot_token_count"],
                    "has_full_snapshot": snapshot["has_full_snapshot"],
                    "fee_known": snapshot["fee_known"],
                    "fee_rate_bps": snapshot["fee_rate_bps"],
                    "quote_valid": replay._valid_reconstructed_quote(
                        snapshot["up_best_bid"], snapshot["up_best_ask"],
                        snapshot["up_best_ask_size_shares"],
                    ) and replay._valid_reconstructed_quote(
                        snapshot["down_best_bid"], snapshot["down_best_ask"],
                        snapshot["down_best_ask_size_shares"],
                    ),
                    "up_best_bid": snapshot["up_best_bid"],
                    "up_best_ask": snapshot["up_best_ask"],
                    "up_best_ask_size_shares": snapshot["up_best_ask_size_shares"],
                    "up_ask_age_seconds": snapshot["up_ask_age_seconds"],
                    "up_ask_received_age_seconds": snapshot["up_ask_received_age_seconds"],
                    "down_best_bid": snapshot["down_best_bid"],
                    "down_best_ask": snapshot["down_best_ask"],
                    "down_best_ask_size_shares": snapshot["down_best_ask_size_shares"],
                    "down_ask_age_seconds": snapshot["down_ask_age_seconds"],
                    "down_ask_received_age_seconds": snapshot["down_ask_received_age_seconds"],
                }
            schedule_position += 1

    state_rows = 0
    rss = _RssSampler()
    rss.start()
    started = time.perf_counter()
    for part_position, path in enumerate(part_paths):
        if part_position < start_position:
            continue
        table = pq.read_table(path, columns=event_columns)
        if table.num_rows:
            market_type = table.schema.field("market").type
            values = [
                condition_id.encode("ascii")
                if pa.types.is_binary(market_type) or pa.types.is_large_binary(market_type)
                else condition_id
                for condition_id in condition_ids
            ]
            selected = table.filter(
                pc.is_in(table["market"], value_set=pa.array(values, type=market_type))
            )
            if selected.num_rows:
                frame = selected.to_pandas()
                frame["market"] = frame.market.map(_as_text)
                frame["_received_ns"] = replay._timestamp_ns(frame.timestamp_received)
                frame["_source_ns"] = replay._timestamp_ns(frame.timestamp)
                row_deadlines = frame.market.map(deadlines)
                frame = frame.loc[frame._received_ns.le(row_deadlines)].copy()
            else:
                frame = pd.DataFrame(columns=[*event_columns, "_received_ns", "_source_ns"])
        else:
            frame = pd.DataFrame(columns=[*event_columns, "_received_ns", "_source_ns"])
        if not frame.empty:
            frame.sort_values(
                ["_received_ns", "_source_ns", "market", "asset_id", "event_type"],
                kind="stable",
                inplace=True,
            )
            for received_ns, receive_group in frame.groupby("_received_ns", sort=False):
                capture_before(int(received_ns), inclusive=False)
                receive_group = receive_group.sort_values(
                    ["_source_ns", "market", "asset_id", "event_type"],
                    kind="stable",
                )
                for condition_id, event_group in receive_group.groupby("market", sort=False):
                    market = market_by_id.get(condition_id)
                    if market is None:
                        continue
                    state = states.setdefault(condition_id, replay._new_state())
                    replay._process_receive_group(
                        state,
                        event_group,
                        int(received_ns),
                        str(market["up_token_id"]),
                        str(market["down_token_id"]),
                    )
                state_rows += len(receive_group)
                capture_before(int(received_ns), inclusive=True)
        partition_end_ns = int(
            (pd.Timestamp(path.stem.replace("T", " "), tz="UTC") + pd.Timedelta(hours=1)).value
        )
        capture_before(partition_end_ns, inclusive=False)
        if checkpoint_enabled and (
            (part_position + 1) % 12 == 0
            or part_position + 1 == len(part_paths)
        ):
            save_checkpoint(part_position, path.name)
            print(
                f"[bbo-refresh] {path.name} partitions={part_position + 1:,}/"
                f"{len(part_paths):,} rows={state_rows:,} snapshots={len(replayed):,} "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
        del table, frame
    capture_before(schedule_times[-1], inclusive=True)
    peak_rss = rss.stop()
    if len(replayed) != len(target_coverage):
        raise RuntimeError(
            f"Targeted BBO replay emitted {len(replayed)} of {len(target_coverage)} requested entries"
        )
    result_frame = pd.DataFrame(replayed.values()).sort_values(
        ["entry_case", "condition_id"], kind="stable"
    )
    if write_outputs:
        result_frame.to_csv(REPORT_DIR / "fresh_bbo_entry_results.csv", index=False)
        _write_json(REPORT_DIR / "fresh_bbo_replay_summary.json", {
            "replay_semantics_version": 3,
            "target_reason_before_replay": "complete direct books and semantically valid quotes at T-60 or T-59",
            "target_entry_cases": [UP_ENTRY_CASE, MAIN_ENTRY_CASE],
            "target_market_count": len(condition_ids),
            "target_fingerprint": target_fingerprint,
            "target_entry_count": len(target_coverage),
            "target_first_event_utc": target_coverage.market_first_event_utc.min().isoformat(),
            "target_last_entry_utc": target_coverage.entry_time_utc.max().isoformat(),
            "partitions_scanned": len(part_paths),
            "rows_applied_to_target_market_states": int(state_rows),
            "elapsed_seconds": float(time.perf_counter() - started),
            "peak_sampled_rss_bytes": peak_rss,
            "rss_sample_interval_ms": 100,
            "network_or_archive_download": False,
            "scope_rationale": "Only markets with complete direct books and semantically valid top quotes can reach or be rejected by the ask-BBO validator. Invalid or missing books fail earlier checks and cannot change eligibility under this freshness correction.",
        })
    return result_frame


def _refresh_quote_validation():
    coverage_path = REPORT_DIR / "data_coverage.csv"
    snapshots_path = REPORT_DIR / "entry_snapshots.parquet"
    coverage = pd.read_csv(coverage_path)
    if "resolved_at_utc" not in coverage.columns:
        resolution = pd.read_parquet(
            snapshots_path,
            columns=["condition_id", "entry_case", "resolved_at_utc"],
        ).drop_duplicates(["condition_id", "entry_case"])
        resolution["condition_id"] = resolution.condition_id.map(_as_text)
        coverage = coverage.merge(
            resolution,
            on=["condition_id", "entry_case"],
            how="left",
            validate="one_to_one",
        )
    target_coverage = coverage.loc[
        coverage.entry_case.isin([UP_ENTRY_CASE, MAIN_ENTRY_CASE])
        & coverage.has_full_snapshot.fillna(False).astype(bool)
        & coverage.quote_valid.fillna(False).astype(bool)
    ].copy()
    if target_coverage.empty:
        raise RuntimeError("No complete semantically valid book snapshots need BBO revalidation")

    replay_code_hash = _sha256(ROOT / "run_btc_preopen_economic_replay.py")
    target_fingerprint, archive_fingerprint, targeted_partition_count = _target_replay_identity(
        target_coverage
    )
    existing_summary_path = REPORT_DIR / "fresh_bbo_replay_summary.json"
    if existing_summary_path.is_file():
        existing_summary = _json(existing_summary_path)
        if (
            existing_summary.get("completed") is True
            and existing_summary.get("replay_semantics_version") == 3
            and existing_summary.get("target_fingerprint") == target_fingerprint
            and existing_summary.get("archive_partition_fingerprint") == archive_fingerprint
            and existing_summary.get("partitions_scanned") == targeted_partition_count
            and existing_summary.get("after_snapshot_hash_sha256") == _sha256(snapshots_path)
            and existing_summary.get("after_coverage_hash_sha256") == _sha256(coverage_path)
        ):
            print("[bbo-refresh] verified saved targeted replay; target, archive, and output fingerprints match", flush=True)
            return existing_summary

    before_columns = [
        "condition_id", "market_slug", "market_start_utc", "entry_case",
        "entry_time_utc", "coverage_reason_age30s", "has_full_snapshot",
        "quote_valid", "quote_valid_strict_bid_lt_ask", "bbo_at_entry_checks",
        "bbo_at_entry_mismatches", "bbo_at_entry_ask_checks",
        "bbo_at_entry_ask_mismatches", "up_best_bid", "up_best_ask",
        "down_best_bid", "down_best_ask", "market_event_count",
    ]
    before_path = REPORT_DIR / "quote_validation_before_fresh_bbo.csv"
    coverage.loc[
        coverage.entry_case.isin([UP_ENTRY_CASE, MAIN_ENTRY_CASE]), before_columns
    ].to_csv(before_path, index=False)

    results = _replay_fresh_bbo_cases(target_coverage, checkpoint_enabled=True)
    old_snapshots = pd.read_parquet(snapshots_path)
    old_snapshots["condition_id"] = old_snapshots.condition_id.map(_as_text)
    keys = ["condition_id", "entry_case"]
    comparisons = old_snapshots.merge(
        results,
        on=keys,
        how="inner",
        suffixes=("_before", "_after"),
        validate="one_to_one",
    )
    if len(comparisons) != len(results):
        raise RuntimeError("Fresh BBO results do not join one-to-one to cached entries")
    for outcome in ("up", "down"):
        for side in ("best_bid", "best_ask"):
            before = pd.to_numeric(comparisons[f"{outcome}_{side}_before"], errors="coerce")
            after = pd.to_numeric(comparisons[f"{outcome}_{side}_after"], errors="coerce")
            if not np.allclose(before, after, atol=1e-8, rtol=0.0, equal_nan=True):
                raise RuntimeError(
                    f"Freshness-only replay changed cached {outcome} {side} quotes"
                )
    for outcome in ("up", "down"):
        for side in (
            "best_ask_size_shares", "ask_age_seconds", "ask_received_age_seconds",
        ):
            before = pd.to_numeric(comparisons[f"{outcome}_{side}_before"], errors="coerce")
            after = pd.to_numeric(comparisons[f"{outcome}_{side}_after"], errors="coerce")
            if not np.allclose(before, after, atol=1e-8, rtol=0.0, equal_nan=True):
                raise RuntimeError(
                    f"Freshness-only replay changed cached {outcome} {side} values"
                )
    event_before = pd.to_numeric(comparisons.market_event_count_before, errors="coerce")
    event_after = pd.to_numeric(comparisons.market_event_count_after, errors="coerce")
    if not event_before.equals(event_after):
        raise RuntimeError("Targeted BBO replay does not reproduce cached pre-entry event counts")
    for column in ("market_book_snapshot_count", "book_snapshot_token_count", "has_full_snapshot", "fee_known"):
        if not comparisons[f"{column}_before"].equals(comparisons[f"{column}_after"]):
            raise RuntimeError(f"Targeted BBO replay changed cached {column}")

    bbo_columns = [
        "bbo_at_entry_checks", "bbo_at_entry_mismatches", "bbo_at_entry_ask_checks",
        "bbo_at_entry_ask_mismatches", "bbo_at_entry_not_newer_than_side_state",
        "bbo_at_entry_older_than_side_state", "bbo_at_entry_tied_with_side_state",
    ]
    updated_snapshots = old_snapshots.copy()
    for column in (
        "bbo_at_entry_not_newer_than_side_state",
        "bbo_at_entry_older_than_side_state",
        "bbo_at_entry_tied_with_side_state",
    ):
        updated_snapshots[column] = pd.NA
    fresh_values = results.set_index(keys)
    cached_index = pd.MultiIndex.from_frame(updated_snapshots[keys])
    target_mask = cached_index.isin(fresh_values.index)
    for column in bbo_columns:
        if column in fresh_values.columns:
            updated_snapshots.loc[target_mask, column] = cached_index[target_mask].map(
                fresh_values[column]
            ).to_numpy()
    temporary_snapshot_path = snapshots_path.with_suffix(".parquet.tmp")
    updated_snapshots.to_parquet(temporary_snapshot_path, index=False)
    os.replace(temporary_snapshot_path, snapshots_path)

    fresh_coverage = results.loc[:, keys + bbo_columns]
    coverage = coverage.merge(
        fresh_coverage,
        on=keys,
        how="left",
        suffixes=("", "_fresh"),
        validate="one_to_one",
    )
    changed = coverage.bbo_at_entry_ask_mismatches_fresh.notna()
    for column in bbo_columns:
        fresh_column = f"{column}_fresh"
        if fresh_column in coverage.columns:
            coverage.loc[changed, column] = coverage.loc[changed, fresh_column]
            coverage.drop(columns=fresh_column, inplace=True)
    coverage["coverage_reason_age30s_before_fresh_bbo"] = coverage.coverage_reason_age30s
    for age in (1, 5, 30):
        reason_col = f"coverage_reason_age{age}s"
        coverage.loc[changed, reason_col] = coverage.loc[changed].apply(
            lambda row: replay._data_reason(row, age), axis=1
        )
    coverage.to_csv(coverage_path, index=False)

    comparison_path = REPORT_DIR / "quote_validation_before_after.csv"
    impact = coverage.loc[
        coverage.entry_case.isin([UP_ENTRY_CASE, MAIN_ENTRY_CASE]),
        [
            "condition_id", "market_slug", "market_start_utc", "entry_case",
            "coverage_reason_age30s_before_fresh_bbo", "coverage_reason_age30s",
            "quote_valid_strict_bid_lt_ask", "quote_valid",
            "bbo_at_entry_ask_mismatches", "bbo_at_entry_not_newer_than_side_state",
            "bbo_at_entry_older_than_side_state", "bbo_at_entry_tied_with_side_state",
        ],
    ].copy()
    impact.to_csv(comparison_path, index=False)
    before_counts = coverage.loc[
        coverage.entry_case.isin([UP_ENTRY_CASE, MAIN_ENTRY_CASE])
    ].groupby("entry_case").coverage_reason_age30s_before_fresh_bbo.value_counts().unstack(fill_value=0)
    after_counts = coverage.loc[
        coverage.entry_case.isin([UP_ENTRY_CASE, MAIN_ENTRY_CASE])
    ].groupby("entry_case").coverage_reason_age30s.value_counts().unstack(fill_value=0)
    changed_rows = impact.loc[
        impact.coverage_reason_age30s_before_fresh_bbo.ne(impact.coverage_reason_age30s)
    ]
    old_target_ask_mismatches = target_coverage.loc[:, [*keys, "bbo_at_entry_ask_mismatches"]].merge(
        results.loc[:, keys + [
            "bbo_at_entry_ask_mismatches",
            "bbo_at_entry_not_newer_than_side_state",
        ]],
        on=keys,
        how="inner",
        suffixes=("_before", "_after"),
        validate="one_to_one",
    )
    old_mismatch_removed_as_stale = (
        pd.to_numeric(old_target_ask_mismatches.bbo_at_entry_ask_mismatches_before, errors="coerce").gt(0)
        & pd.to_numeric(old_target_ask_mismatches.bbo_at_entry_ask_mismatches_after, errors="coerce").eq(0)
        & pd.to_numeric(old_target_ask_mismatches.bbo_at_entry_not_newer_than_side_state, errors="coerce").gt(0)
    )

    def _reason_counts_by_case(counts):
        return {
            str(case): {str(reason): int(value) for reason, value in row.items()}
            for case, row in counts.iterrows()
        }

    summary = _json(REPORT_DIR / "fresh_bbo_replay_summary.json")
    summary.update({
        "eligible_before_by_entry_case": {
            case: int(values.get("eligible", 0)) for case, values in before_counts.iterrows()
        },
        "eligible_after_by_entry_case": {
            case: int(values.get("eligible", 0)) for case, values in after_counts.iterrows()
        },
        "reason_counts_before_by_entry_case": _reason_counts_by_case(before_counts),
        "reason_counts_after_by_entry_case": _reason_counts_by_case(after_counts),
        "primary_reason_changed_market_entry_rows": int(len(changed_rows)),
        "newly_eligible_market_entry_rows": int((
            changed_rows.coverage_reason_age30s.eq("eligible")
        ).sum()),
        "ask_mismatches_rejected_as_old_or_tied_references": int(
            old_mismatch_removed_as_stale.sum()
        ),
        "ask_mismatches_still_confirmed_by_fresh_reference": int(
            results.bbo_at_entry_ask_mismatches.sum()
        ),
        "before_snapshot_hash_sha256": _sha256(before_path),
        "after_snapshot_hash_sha256": _sha256(snapshots_path),
        "after_coverage_hash_sha256": _sha256(coverage_path),
        "replay_code_sha256": replay_code_hash,
        "completed": True,
        "before_coverage_path": before_path.relative_to(ROOT).as_posix(),
        "after_coverage_path": coverage_path.relative_to(ROOT).as_posix(),
        "entry_impact_path": comparison_path.relative_to(ROOT).as_posix(),
        "results_path": "reports/btc_preopen/fresh_bbo_entry_results.csv",
    })
    _write_json(REPORT_DIR / "fresh_bbo_replay_summary.json", summary)
    return summary


def _build_book_flags():
    coverage = pd.read_csv(REPORT_DIR / "data_coverage.csv")
    quote_sources = pd.read_parquet(
        REPORT_DIR / "entry_snapshots.parquet",
        columns=[
            "condition_id", "entry_case", "up_quote_source_token_id",
            "down_quote_source_token_id",
        ],
    ).drop_duplicates(["condition_id", "entry_case"])
    coverage = coverage.merge(
        quote_sources,
        on=["condition_id", "entry_case"],
        how="left",
        validate="one_to_one",
    )
    market_index = pd.read_parquet(
        ARCHIVE_DIR / "market_index.parquet",
        columns=["condition_id", "up_token_id", "down_token_id"],
    )
    market_index["condition_id"] = market_index.condition_id.map(_as_text)
    market_index["up_token_id"] = market_index.up_token_id.map(_as_text)
    market_index["down_token_id"] = market_index.down_token_id.map(_as_text)
    frame = coverage.loc[
        coverage.entry_case.isin([UP_ENTRY_CASE, MAIN_ENTRY_CASE])
    ].merge(market_index, on="condition_id", how="left", validate="many_to_one")

    count = pd.to_numeric(frame.book_snapshot_token_count, errors="coerce")
    up_source = frame.up_quote_source_token_id.fillna("").astype(str)
    down_source = frame.down_quote_source_token_id.fillna("").astype(str)
    no_initial_up = count.eq(0) | (count.eq(1) & up_source.ne(frame.up_token_id))
    no_initial_down = count.eq(0) | (count.eq(1) & down_source.ne(frame.down_token_id))

    for outcome in ("up", "down"):
        for side in ("bid", "ask"):
            frame[f"_{outcome}_{side}"] = pd.to_numeric(frame[f"{outcome}_best_{side}"], errors="coerce")
    any_missing_top = frame[[f"_{outcome}_{side}" for outcome in ("up", "down") for side in ("bid", "ask")]].isna().any(axis=1)
    any_crossed = (
        (frame._up_bid > frame._up_ask)
        | (frame._down_bid > frame._down_ask)
    ).fillna(False)
    any_locked = (
        (frame._up_bid == frame._up_ask)
        | (frame._down_bid == frame._down_ask)
    ).fillna(False)
    any_bad_price = pd.Series(False, index=frame.index)
    for outcome in ("up", "down"):
        for side in ("bid", "ask"):
            price = frame[f"_{outcome}_{side}"]
            any_bad_price |= price.notna() & ((price <= 0.0) | (price >= 1.0))

    frame["missing_up_initial_book"] = no_initial_up.astype(bool)
    frame["missing_down_initial_book"] = no_initial_down.astype(bool)
    frame["missing_required_book_side"] = any_missing_top.astype(bool)
    frame["ask_bbo_mismatch"] = pd.to_numeric(frame.bbo_at_entry_ask_mismatches, errors="coerce").fillna(0).gt(0)
    frame["bid_only_bbo_mismatch"] = (
        pd.to_numeric(frame.bbo_at_entry_mismatches, errors="coerce").fillna(0)
        > pd.to_numeric(frame.bbo_at_entry_ask_mismatches, errors="coerce").fillna(0)
    )
    frame["crossed_book_bid_gt_ask"] = any_crossed.astype(bool)
    frame["locked_book_bid_eq_ask"] = any_locked.astype(bool)
    frame["invalid_best_price_level"] = any_bad_price.astype(bool)
    frame["missing_or_unchecked_reference_bbo"] = pd.to_numeric(frame.bbo_at_entry_checks, errors="coerce").fillna(0).lt(2)
    side_time_check_count = pd.to_numeric(
        frame.bbo_at_entry_not_newer_than_side_state, errors="coerce"
    )
    frame["reference_bbo_not_newer_than_both_side_updates"] = pd.Series(
        pd.NA, index=frame.index, dtype="boolean"
    )
    checked_rows = side_time_check_count.notna()
    frame.loc[checked_rows, "reference_bbo_not_newer_than_both_side_updates"] = (
        side_time_check_count.loc[checked_rows].gt(0)
    )
    frame["stale_ask_change_age_over_30s"] = (
        pd.to_numeric(frame.up_ask_age_seconds, errors="coerce").isna()
        | pd.to_numeric(frame.down_ask_age_seconds, errors="coerce").isna()
        | pd.concat([
            pd.to_numeric(frame.up_ask_age_seconds, errors="coerce"),
            pd.to_numeric(frame.down_ask_age_seconds, errors="coerce"),
        ], axis=1).max(axis=1).gt(ASK_AGE_SECONDS)
    )
    receive_ages = pd.concat([
        pd.to_numeric(frame.up_ask_received_age_seconds, errors="coerce"),
        pd.to_numeric(frame.down_ask_received_age_seconds, errors="coerce"),
    ], axis=1)
    frame["ask_receive_age_over_30s"] = (
        receive_ages.isna().any(axis=1)
        | receive_ages.max(axis=1).gt(ASK_AGE_SECONDS)
    )
    frame["unknown_historical_fees"] = ~frame.fee_known.fillna(False).astype(bool)
    frame["insufficient_5usd_native_ask_depth"] = ~frame.depth_5usd_valid_both_sides.fillna(False).astype(bool)
    frame["future_receive_contamination"] = ~frame.no_future_event_at_entry.fillna(False).astype(bool)
    frame["future_source_timestamp"] = ~frame.no_future_source_event_at_entry.fillna(False).astype(bool)
    frame["maintenance_pause"] = frame.fee_collection_mode.eq("maintenance_pause")
    frame["uninitialized_price_changes_before_entry"] = pd.to_numeric(
        frame.uninitialized_price_change_count, errors="coerce"
    ).fillna(0).gt(0)
    frame["archive_event_sequence_id_available"] = False
    frame["same_timestamp_order_ambiguity_count"] = pd.to_numeric(
        frame.bbo_at_entry_tied_with_side_state, errors="coerce"
    )
    frame["reference_bbo_side_timestamp_comparison_available"] = checked_rows
    frame["invalid_raw_levels_count_available"] = False
    frame["primary_reason"] = frame.coverage_reason_age30s.astype(str)
    frame["strategy_skip_or_balance_gate"] = False
    frame["feed_sequence_or_connection_continuity_available"] = False
    frame["market_start_date_utc"] = _date_key(frame.market_start_utc)

    flags = [
        "missing_up_initial_book", "missing_down_initial_book", "missing_required_book_side",
        "ask_bbo_mismatch", "bid_only_bbo_mismatch", "crossed_book_bid_gt_ask",
        "locked_book_bid_eq_ask", "invalid_best_price_level", "missing_or_unchecked_reference_bbo",
        "reference_bbo_not_newer_than_both_side_updates",
        "stale_ask_change_age_over_30s", "ask_receive_age_over_30s", "unknown_historical_fees",
        "insufficient_5usd_native_ask_depth", "future_receive_contamination", "future_source_timestamp",
        "maintenance_pause", "uninitialized_price_changes_before_entry",
    ]
    output_columns = [
        "condition_id", "market_slug", "market_start_utc", "market_start_date_utc", "entry_case",
        "entry_time_utc", "market_first_event_utc", "market_last_event_utc", "market_event_count",
        "market_book_snapshot_count", "primary_reason", "has_full_snapshot", "book_snapshot_token_count",
        "quote_valid", "quote_valid_strict_bid_lt_ask", "up_best_bid", "up_best_ask",
        "down_best_bid", "down_best_ask", "up_ask_age_seconds", "up_ask_received_age_seconds",
        "down_ask_age_seconds", "down_ask_received_age_seconds", "bbo_at_entry_checks",
        "bbo_at_entry_mismatches", "bbo_at_entry_ask_mismatches", "uninitialized_price_change_count",
        "bbo_at_entry_not_newer_than_side_state", "bbo_at_entry_older_than_side_state",
        "bbo_at_entry_tied_with_side_state",
        *flags, "archive_event_sequence_id_available", "same_timestamp_order_ambiguity_count",
        "reference_bbo_side_timestamp_comparison_available", "invalid_raw_levels_count_available",
        "strategy_skip_or_balance_gate", "feed_sequence_or_connection_continuity_available",
    ]
    output = REPORT_DIR / "book_rejection_markets.csv"
    frame.loc[:, output_columns].to_csv(output, index=False)

    daily_rows = []
    for (case, date, reason), count_value in frame.groupby(
            ["entry_case", "market_start_date_utc", "primary_reason"],
            dropna=False,
    ).size().items():
        daily_rows.append({
            "entry_case": case,
            "market_start_date_utc": date,
            "metric_type": "primary_reason",
            "metric": reason,
            "markets": int(count_value),
        })
    for flag in flags:
        counts = frame.groupby(["entry_case", "market_start_date_utc"], dropna=False)[flag].sum()
        for (case, date), count_value in counts.items():
            daily_rows.append({
                "entry_case": case,
                "market_start_date_utc": date,
                "metric_type": "overlapping_flag",
                "metric": flag,
                "markets": int(count_value),
            })
    pd.DataFrame(daily_rows).to_csv(REPORT_DIR / "book_rejection_daily.csv", index=False)

    summary = {}
    for case, group in frame.groupby("entry_case"):
        primary_counts = group.primary_reason.value_counts().astype(int).to_dict()
        summary[case] = {
            "markets": int(len(group)),
            "primary_reason_counts_sum_to_markets": int(sum(primary_counts.values())) == len(group),
            "primary_reason_counts": primary_counts,
            "overlapping_flag_counts": {flag: int(group[flag].sum()) for flag in flags},
            "strict_bid_lt_ask_quotes": int(group.quote_valid_strict_bid_lt_ask.sum()),
            "semantically_valid_quotes_bid_lte_ask": int(group.quote_valid.sum()),
            "eligible_markets": int((group.primary_reason == "eligible").sum()),
            "entry_reference_bbo_source_time_relations": {
                "checked_market_entries": int(group.bbo_at_entry_not_newer_than_side_state.notna().sum()),
                "older_reference_events": int(pd.to_numeric(
                    group.bbo_at_entry_older_than_side_state, errors="coerce"
                ).fillna(0).sum()),
                "tied_reference_events": int(pd.to_numeric(
                    group.bbo_at_entry_tied_with_side_state, errors="coerce"
                ).fillna(0).sum()),
            },
        }
    return frame, summary


def _event_rows_for_market(condition_id, first_event_time, entry_time):
    columns = [
        "timestamp_received", "timestamp", "market", "event_type", "asset_id",
        "bids", "asks", "price", "size", "side", "best_bid", "best_ask", "fee_rate_bps",
    ]
    selected = []
    first_hour = pd.Timestamp(first_event_time).floor("h")
    last_hour = pd.Timestamp(entry_time).floor("h")
    for path in sorted(PARTS_DIR.glob("*.parquet")):
        partition_hour = pd.Timestamp(path.stem.replace("T", " "), tz="UTC")
        if partition_hour < first_hour or partition_hour > last_hour:
            continue
        parquet = pq.ParquetFile(path)
        market_index = parquet.schema.names.index("market")
        for row_group_index in range(parquet.num_row_groups):
            metadata = parquet.metadata.row_group(row_group_index).column(market_index)
            stats = metadata.statistics
            if stats is not None and stats.has_min_max:
                minimum, maximum = _as_text(stats.min), _as_text(stats.max)
                if condition_id < minimum or condition_id > maximum:
                    continue
            table = parquet.read_row_group(row_group_index, columns=columns)
            if table.num_rows == 0:
                continue
            part = table.to_pandas()
            part["market"] = part["market"].map(_as_text)
            selected.append(part.loc[part.market.eq(condition_id)])
    parts = [part for part in selected if not part.empty]
    if not parts:
        raise RuntimeError(f"No local PMXT events found for {condition_id}")
    frame = pd.concat(parts, ignore_index=True)
    frame["timestamp_received"] = pd.to_datetime(frame.timestamp_received, utc=True, errors="coerce")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce")
    frame["asset_id"] = frame.asset_id.map(_as_text)
    frame["_received_ns"] = replay._timestamp_ns(frame.timestamp_received)
    # Match the replay's stable source sort exactly: missing source timestamps
    # stay at pandas' NaT int64 sentinel, while _update_event falls back to receive time.
    frame["_source_ns"] = replay._timestamp_ns(frame.timestamp)
    frame.sort_values(
        ["_received_ns", "_source_ns", "market", "asset_id", "event_type"],
        kind="stable",
        inplace=True,
    )
    return frame


def _top_pair(state, up_token_id, down_token_id):
    output = {}
    for name, token_id in (("up", up_token_id), ("down", down_token_id)):
        book = replay._direct_book(state["tokens"].get(str(token_id)))
        top = replay._book_top(book)
        output[name] = None if top is None else {
            "bid": float(top[0]),
            "ask": float(top[1]),
            "bid_size": float(top[2]),
            "ask_size": float(top[3]),
        }
    return output


def _trace_market(row, market):
    condition_id = str(row.condition_id)
    up_token_id, down_token_id = str(market.up_token_id), str(market.down_token_id)
    entry = pd.Timestamp(row.entry_time_utc)
    frame = _event_rows_for_market(
        condition_id,
        row.market_first_event_utc,
        entry,
    )
    frame = frame.loc[frame._received_ns.le(entry.value)].copy()
    state = replay._new_state()
    crossed_example = None
    ask_mismatch_example = None
    stale_or_tied_ask_reference_example = None
    for received_ns, receive_group in frame.groupby("_received_ns", sort=False):
        receive_group = receive_group.sort_values(
            ["_source_ns", "market", "asset_id", "event_type"],
            kind="stable",
        )
        before = _top_pair(state, up_token_id, down_token_id)
        before_cross = any(
            quote is not None and quote["bid"] > quote["ask"]
            for quote in before.values()
        )
        event_rows = []
        for event_row in receive_group.itertuples(index=False):
            if event_row.event_type in {"book", "price_change"}:
                event_rows.append({
                    "event_type": str(event_row.event_type),
                    "token_id": str(event_row.asset_id),
                    "source_at_utc": pd.Timestamp(event_row.timestamp).isoformat(),
                    "side": None if pd.isna(event_row.side) else str(event_row.side),
                    "price": None if pd.isna(event_row.price) else float(event_row.price),
                    "size": None if pd.isna(event_row.size) else float(event_row.size),
                    "reported_bbo": [
                        None if pd.isna(event_row.best_bid) else float(event_row.best_bid),
                        None if pd.isna(event_row.best_ask) else float(event_row.best_ask),
                    ],
                    "full_book_level_counts": [
                        len(replay._levels(event_row.bids)),
                        len(replay._levels(event_row.asks)),
                    ] if event_row.event_type == "book" else None,
                })
        replay._process_receive_group(
            state,
            receive_group,
            int(received_ns),
            up_token_id,
            down_token_id,
        )
        after = _top_pair(state, up_token_id, down_token_id)
        after_cross = any(
            quote is not None and quote["bid"] > quote["ask"]
            for quote in after.values()
        )
        base = {
            "received_at_utc": pd.Timestamp(received_ns, unit="ns", tz="UTC").isoformat(),
            "before": before,
            "messages": event_rows,
            "after": after,
        }
        if crossed_example is None and after_cross and not before_cross:
            crossed_example = base
        for token_id in (up_token_id, down_token_id):
            token = state["tokens"].get(token_id)
            if token is None or token["reported_receive_ns"] != int(received_ns):
                continue
            source_ns = token["reported_source_ns"]
            side_sources = (token["bid_book_source_ns"], token["ask_book_source_ns"])
            latest_side_source_ns = max(
                value for value in side_sources if value is not None
            ) if any(value is not None for value in side_sources) else None
            actual = replay._bbo(replay._direct_book(token))
            if source_ns is None or latest_side_source_ns is None or actual is None:
                continue
            sample = {
                "token_id": token_id,
                "received_at_utc": pd.Timestamp(received_ns, unit="ns", tz="UTC").isoformat(),
                "source_timestamp_utc": pd.Timestamp(source_ns, unit="ns", tz="UTC").isoformat(),
                "latest_side_source_timestamp_utc": pd.Timestamp(
                    latest_side_source_ns, unit="ns", tz="UTC"
                ).isoformat(),
                "relation": (
                    "older" if source_ns < latest_side_source_ns
                    else "tied" if source_ns == latest_side_source_ns
                    else "newer"
                ),
                "reported_bbo": [token["reported_best_bid"], token["reported_best_ask"]],
                "reconstructed_bbo": [actual[0], actual[1]],
            }
            if abs(sample["reported_bbo"][1] - actual[1]) <= 1e-6:
                continue
            if source_ns <= latest_side_source_ns:
                stale_or_tied_ask_reference_example = {
                    **base,
                    "bbo_check": sample,
                }
            else:
                ask_mismatch_example = {**base, "bbo_check": sample}

    final_top = _top_pair(state, up_token_id, down_token_id)
    for outcome in ("up", "down"):
        cached_bid, cached_ask = row[f"{outcome}_best_bid"], row[f"{outcome}_best_ask"]
        reconstructed = final_top[outcome]
        if reconstructed is not None and (
            abs(reconstructed["bid"] - float(cached_bid)) > 1e-8
            or abs(reconstructed["ask"] - float(cached_ask)) > 1e-8
        ):
            raise RuntimeError(
                "Targeted PMXT trace does not reproduce the cached entry top: "
                f"{condition_id} {outcome}: trace={reconstructed}, "
                f"cache=({cached_bid}, {cached_ask}), "
                f"trace_events={len(frame)}, cached_events={row.market_event_count}"
            )
    return {
        "condition_id": condition_id,
        "market_slug": str(row.market_slug),
        "entry_time_utc": entry.isoformat(),
        "tokens": {"up": up_token_id, "down": down_token_id},
        "event_rows_before_entry": int(len(frame)),
        "reconstructed_entry_top": final_top,
        "crossing_transition": crossed_example,
        "reported_ask_mismatch": ask_mismatch_example,
        "stale_or_tied_ask_reference": stale_or_tied_ask_reference_example,
    }


def _trace_markets(coverage_frame):
    market_index = pd.read_parquet(
        ARCHIVE_DIR / "market_index.parquet",
        columns=["condition_id", "up_token_id", "down_token_id"],
    )
    market_index["condition_id"] = market_index.condition_id.map(_as_text)
    main = coverage_frame.loc[coverage_frame.entry_case.eq(MAIN_ENTRY_CASE)].copy()
    top_missing = main[["up_best_bid", "up_best_ask", "down_best_bid", "down_best_ask"]].isna().any(axis=1)
    up_bid = pd.to_numeric(main.up_best_bid, errors="coerce")
    up_ask = pd.to_numeric(main.up_best_ask, errors="coerce")
    down_bid = pd.to_numeric(main.down_best_bid, errors="coerce")
    down_ask = pd.to_numeric(main.down_best_ask, errors="coerce")
    cross_mask = ((up_bid > up_ask) | (down_bid > down_ask)).fillna(False) & ~top_missing
    sort_columns = ["market_start_utc", "condition_id"]
    cross_row = main.loc[cross_mask].sort_values(sort_columns).iloc[0]
    ask_mismatch = pd.to_numeric(
        main.bbo_at_entry_ask_mismatches, errors="coerce"
    ).fillna(0).gt(0)
    ask_rows = main.loc[ask_mismatch & ~cross_mask]
    if not ask_rows.empty:
        ask_row = ask_rows.sort_values(sort_columns).iloc[0]
    else:
        ask_rows = main.loc[ask_mismatch]
        ask_row = ask_rows.sort_values(sort_columns).iloc[0]
    not_newer = pd.to_numeric(
        main.bbo_at_entry_not_newer_than_side_state, errors="coerce"
    ).fillna(0).gt(0)
    stale_rows = main.loc[not_newer & ~cross_mask & ~ask_mismatch]
    if stale_rows.empty:
        stale_rows = main.loc[not_newer & ~cross_mask]
    stale_row = stale_rows.sort_values(sort_columns).iloc[0]
    trace_targets = []
    for purpose, row in (
        ("crossed book reconstruction", cross_row),
        ("fresh ask BBO disagreement", ask_row),
        ("stale or tied BBO must not decide", stale_row),
    ):
        if any(str(row.condition_id) == str(existing.condition_id) for _, existing in trace_targets):
            continue
        trace_targets.append((purpose, row))
    row_payloads = []
    for purpose, row in trace_targets:
        joined = market_index.loc[market_index.condition_id.eq(str(row.condition_id))]
        if len(joined) != 1:
            raise RuntimeError(f"Missing market token map for {row.condition_id}")
        trace = _trace_market(row, joined.iloc[0])
        trace["trace_purpose"] = purpose
        trace["coverage_reason"] = str(row.coverage_reason_age30s)
        trace["quote_valid_strict_bid_lt_ask"] = bool(row.quote_valid_strict_bid_lt_ask)
        trace["quote_valid_bid_lte_ask"] = bool(row.quote_valid)
        row_payloads.append(trace)
    output_lines = [
        "# PMXT book reconstruction traces",
        "",
        "The traces use only cached PMXT rows for the listed market and received no later than its T−59 entry. Rows are ordered with the replay's receive/source/token/type sort, then compared after the full receive group. PMXT v2 does not publish a monotonic event sequence ID, so exact timestamp ties remain unresolved.",
        "",
        "A locked quote has a non-empty native bid and ask at the same valid price; a BUY at the ask remains defined. The corrected validator accepts `bid == ask` and still rejects `bid > ask`. Reconciliation mismatches remain excluded from the economic scenario.",
        "A reported BBO is compared only when its source timestamp is strictly newer than the latest source timestamp for both native side states. Older and tied references are counted separately; PMXT lacks the event sequence needed to order ties.",
        "",
    ]
    for trace in row_payloads:
        output_lines.extend([
            f"## {trace['trace_purpose']}: {trace['market_slug']} — {trace['condition_id']}",
            "",
            f"Entry: `{trace['entry_time_utc']}`. Cached primary reason: `{trace['coverage_reason']}`. Direct UP/DOWN token IDs are `{trace['tokens']['up']}` / `{trace['tokens']['down']}`.",
            "",
            f"Targeted source scan applied {trace['event_rows_before_entry']:,} market events received by entry. Reconstructed entry top: `{json.dumps(trace['reconstructed_entry_top'], separators=(',', ':'))}`.",
            "",
        ])
        crossing = trace["crossing_transition"]
        if crossing:
            output_lines.extend([
                "First transition into a crossed direct book:",
                "",
                f"- Before: `{json.dumps(crossing['before'], separators=(',', ':'))}`",
                f"- Receive group: `{crossing['received_at_utc']}`",
                f"- Source messages: `{json.dumps(crossing['messages'], separators=(',', ':'))}`",
                f"- After the complete group: `{json.dumps(crossing['after'], separators=(',', ':'))}`",
                "- Interpretation: local depth deltas and the reported BBO disagree on subsequent snapshots. Because the archive omits sequence/removal completeness evidence, this does not establish whether the exporter omitted a level deletion or the messages share an unresolved order. It is not evidence that the exchange had no liquidity.",
                "",
            ])
        mismatch = trace["reported_ask_mismatch"]
        if mismatch:
            output_lines.extend([
                "A received group with a reported ask-BBO mismatch:",
                "",
                f"- Before: `{json.dumps(mismatch['before'], separators=(',', ':'))}`",
                f"- Receive group: `{mismatch['received_at_utc']}`",
                f"- Source messages: `{json.dumps(mismatch['messages'], separators=(',', ':'))}`",
                f"- After the complete group: `{json.dumps(mismatch['after'], separators=(',', ':'))}`",
                f"- Reference check: `{json.dumps(mismatch['bbo_check'], separators=(',', ':'))}`",
                "- Interpretation: the token ID matches, the reference is newer than both side states, and the local reconstructed ask differs from the reported ask. The archive cannot establish whether the difference reflects an omitted depth update or incomplete archived history, so the ask remains excluded.",
                "",
            ])
        unresolved = trace["stale_or_tied_ask_reference"]
        if unresolved:
            output_lines.extend([
                "A reported ask BBO whose source timestamp is not newer than both side states:",
                "",
                f"- Before: `{json.dumps(unresolved['before'], separators=(',', ':'))}`",
                f"- Receive group: `{unresolved['received_at_utc']}`",
                f"- Source messages: `{json.dumps(unresolved['messages'], separators=(',', ':'))}`",
                f"- After the complete group: `{json.dumps(unresolved['after'], separators=(',', ':'))}`",
                f"- Reference check: `{json.dumps(unresolved['bbo_check'], separators=(',', ':'))}`",
                "- Interpretation: the reference is older than or tied with a side update, so its reported BBO cannot be used to declare a fresh disagreement. A tie also has no resolvable sequence order in PMXT.",
                "",
            ])
        if not crossing and not mismatch and not unresolved:
            output_lines.extend([
                "No earlier crossing transition or ask-mismatch group was retained for this market; the final cached top was still reproduced from the local source rows.",
                "",
            ])
    output_lines.extend([
        "## Control semantics and limits",
        "",
        "The replay applies every row in a receive-time group before comparing BBO. A reference is decisive only for the same native token, if received by entry, and if its source timestamp is strictly newer than both reconstructed side-state timestamps at that point. It does not compare a reference to a later full snapshot. PMXT has no sequence ID for exact ties, so their order remains unknown; the side timestamps permit the strict freshness gate but cannot prove the exchange's internal order.",
        "",
        "Price-change size ≤0 removes the price level; identical repeated sizes do not refresh ask-change age; a full `book` replaces both sides and resets their source/receive age clocks. Prices are rounded to 8 decimal places and levels outside (0,1), non-finite levels, and non-positive snapshot sizes are discarded. The saved entry snapshot does not retain counts of discarded raw levels, so it cannot identify whether a particular rejection involved one.",
        "",
    ])
    path = REPORT_DIR / "BOOK_TRACE_EXAMPLES.md"
    path.write_text("\n".join(output_lines), encoding="utf-8")
    return row_payloads


def _artifact_manifest(feature_columns, original_columns):
    feature_sources = _json(MODEL_STAGE_DIR / "feature_sources.json")
    model_bundle = _json(MODEL_STAGE_DIR / "model_bundle.json")
    original_training = _json(MODEL_STAGE_DIR / "training_manifest.json")
    candidate_metrics = _json(REPORT_DIR / "candidate_metrics.json")
    candidate_model = ROOT / candidate_metrics["candidate_model_path"]
    original_model = ROOT / original_training["model_path"]
    active_runtime = _json(ROOT / "configs/runtime/active.json")
    active_model_meta = ROOT / active_runtime["assets"]["BTC"]["artifacts"]["model_meta_path"]
    collector_model_meta = ROOT / "data/models/BTC/btc_preopen_v1/lgbm_meta.json"
    original_calibrator = RUN_DIR / "stages/calibration_6942659d3b63/platt_calibrator.json"
    candidate_calibrator = candidate_model.with_name("candidate_calibrator.json")

    paths = [
        CANDIDATE_RUNTIME,
        CANDIDATE_META,
        CANDIDATE_FEATURE_CONFIG,
        CANDIDATE_HISTORY,
        ROOT / "configs/runtime/btc_preopen_candidate_indicator_state.json",
        active_model_meta,
        collector_model_meta,
        candidate_model,
        candidate_calibrator,
        INDICATOR_FIT_DIR / "BollingerBands_target_5m_preopen_v1_pop64_qe0.2_qm0.1_tf0.8_stmc_sg10_rwlin1-1.5.json",
        INDICATOR_FIT_DIR / "ChaikinOsc_target_5m_preopen_v1_pop64_qe0.2_qm0.1_tf0.8_stmc_sg10_rwlin1-1.5.json",
        INDICATOR_FIT_DIR / "KeltnerChannel_target_5m_preopen_v1_pop64_qe0.2_qm0.1_tf0.8_stmc_sg10_rwlin1-1.5.json",
        INDICATOR_FIT_DIR / "StochOsc_target_5m_preopen_v1_pop64_qe0.2_qm0.1_tf0.8_stmc_sg10_rwlin1-1.5.json",
        PROFILE_STATE_DIR / "volume_profile/BTCUSD_INDEXVOL_UM_BTCUSDT_1m_vp_fixed_range_v3_modeling_end.npz",
        PROFILE_STATE_DIR / "volume_profile/BTCUSD_INDEXVOL_UM_BTCUSDT_1m_vp_fixed_range_v3_modeling_end.json",
        PROFILE_STATE_DIR / "reaction_profile/BTCUSD_INDEXVOL_UM_BTCUSDT_1m_rp_fixed_grid_v1_modeling_end.npz",
        PROFILE_STATE_DIR / "reaction_profile/BTCUSD_INDEXVOL_UM_BTCUSDT_1m_rp_fixed_grid_v1_modeling_end.json",
        original_model,
        original_calibrator,
        ROOT / "configs/btc_preopen_v1.json",
        ROOT / "configs/runtime/trade_policy_project.json",
        RAW_PATH,
        FEATURE_PARQUET,
        MODEL_STAGE_DIR / "feature_order.json",
        MODEL_STAGE_DIR / "feature_sources.json",
        ARCHIVE_DIR / "archive_identity.json",
        ARCHIVE_DIR / "mapping_identity.json",
        ARCHIVE_DIR / "market_index.parquet",
        REPORT_DIR / "entry_snapshots.parquet",
        REPORT_DIR / "event_replay_summary.json",
        REPORT_DIR / "economic_replay.json",
        ROOT / "run_btc_preopen_economic_replay.py",
        ROOT / "audit_btc_preopen_runtime_compatibility.py",
        ROOT / "run.py",
        ROOT / "features/live_indicator_runtime.py",
        ROOT / "utils/project_config.py",
        ROOT / "utils/polymarket_user_stream.py",
        ROOT / "docs/live_telemetry.md",
    ]
    artifact_rows = []
    for path in paths:
        relative = Path(path).relative_to(ROOT)
        if not Path(path).is_file():
            artifact_rows.append({"path": relative.as_posix(), "exists": False})
            continue
        artifact_rows.append({
            "path": relative.as_posix(),
            "exists": True,
            "size_bytes": Path(path).stat().st_size,
            "sha256": _sha256(path),
        })

    partition_rows = []
    partition_schemas = {}
    for path in sorted(PARTS_DIR.glob("*.parquet")):
        schema = pq.ParquetFile(path).schema_arrow
        schema_fields = [
            {"name": field.name, "type": str(field.type)}
            for field in schema
        ]
        schema_hash = hashlib.sha256(
            json.dumps(schema_fields, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        partition_schemas.setdefault(schema_hash, schema_fields)
        partition_rows.append({
            "partition": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "schema_sha256": schema_hash,
        })
    if len(partition_rows) != 811:
        raise RuntimeError(f"Expected 811 cached PMXT partitions, found {len(partition_rows)}")
    pd.DataFrame(partition_rows).to_csv(REPORT_DIR / "archive_partition_hashes.csv", index=False)
    partition_digest = hashlib.sha256(
        "\n".join(f"{row['partition']}:{row['sha256']}" for row in partition_rows).encode()
    ).hexdigest()

    candidate_ids = list(feature_columns)
    manifest = {
        "repository_head_before_this_change": _git_head(),
        "candidate": {
            "feature_count": len(candidate_ids),
            "ordered_feature_sha256_newline_utf8": _sha256_order(candidate_ids),
            "ordered_feature_sha256_json_compact_utf8": hashlib.sha256(
                json.dumps(candidate_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "feature_order_matches_original_v1": candidate_ids == original_columns,
            "model_path": str(candidate_model.relative_to(ROOT)).replace("\\", "/"),
            "calibrator_path": str(candidate_calibrator.relative_to(ROOT)).replace("\\", "/"),
            "model_iterations": int(candidate_metrics.get("iterations", 0)),
            "model_hash_from_candidate_study": candidate_metrics.get("candidate_model_sha256"),
            "indicator_state_seed_path": "configs/runtime/btc_preopen_candidate_indicator_state.json",
            "indicator_state_seed_sha256": (
                _sha256(ROOT / "configs/runtime/btc_preopen_candidate_indicator_state.json")
                if (ROOT / "configs/runtime/btc_preopen_candidate_indicator_state.json").is_file()
                else None
            ),
            "calibration": _json(candidate_calibrator),
            "training_feature_sources_path": feature_sources.get("raw_candle_path"),
            "feature_dataset_path": str(FEATURE_PARQUET.relative_to(ROOT)).replace("\\", "/"),
            "feature_dataset_rows": int(pq.ParquetFile(FEATURE_PARQUET).metadata.num_rows),
        },
        "original_v1": {
            "feature_count": len(original_columns),
            "ordered_feature_sha256_newline_utf8": _sha256_order(original_columns),
            "model_path": str(original_model.relative_to(ROOT)).replace("\\", "/"),
            "model_sha256": original_training.get("model_sha256"),
            "calibrator_path": str(original_calibrator.relative_to(ROOT)).replace("\\", "/"),
        },
        "economics": {
            "entry_case": MAIN_ENTRY_CASE,
            "entry_relative_to_market_start": "T-59s",
            "age_cap_seconds": ASK_AGE_SECONDS,
            "gross_order_usd": replay.GROSS_ORDER_USD,
            "initial_cash_usd": replay.INITIAL_CASH_USD,
            "capital_release_delay_seconds": RELEASE_DELAY_SECONDS,
            "decision_and_entry_config_path": "configs/btc_preopen_v1.json",
            "strategy_policy_config_path": "configs/runtime/trade_policy_project.json",
            "replay_code_path": "run_btc_preopen_economic_replay.py",
        },
        "pmxt_cache": {
            "archive_identity": _json(ARCHIVE_DIR / "archive_identity.json"),
            "mapping_identity": _json(ARCHIVE_DIR / "mapping_identity.json"),
            "partition_count": len(partition_rows),
            "partition_list_hash_sha256": partition_digest,
            "event_type_counts": _json(REPORT_DIR / "event_replay_summary.json").get("event_type_counts"),
            "event_rows": _json(REPORT_DIR / "event_replay_summary.json").get("event_rows"),
            "schema_sequence_id_present": False,
            "schema_version_column_present": any(
                field["name"].lower() == "schema_version"
                for fields in partition_schemas.values()
                for field in fields
            ),
            "observed_partition_schema_count": len(partition_schemas),
            "observed_partition_schemas": [
                {"sha256": digest, "fields": fields}
                for digest, fields in sorted(partition_schemas.items())
            ],
        },
        "artifacts": artifact_rows,
    }
    _write_json(REPORT_DIR / "artifact_manifest.json", manifest)
    return manifest


def _feature_change_audit(candidate_columns, original_columns):
    source_manifest = _json(MODEL_STAGE_DIR / "feature_sources.json")
    model_bundle = _json(MODEL_STAGE_DIR / "model_bundle.json")
    feature_config = _json(CANDIDATE_FEATURE_CONFIG)
    candidate_runtime = _json(CANDIDATE_RUNTIME)["assets"]["BTC"]["artifacts"]
    active_runtime = _json(ROOT / "configs/runtime/active.json")
    active_meta_path = ROOT / active_runtime["assets"]["BTC"]["artifacts"]["model_meta_path"]
    active_meta = _json(active_meta_path)
    active_columns = list(active_meta["feature_columns"])
    collector_meta_path = ROOT / "data/models/BTC/btc_preopen_v1/lgbm_meta.json"
    collector_meta = _json(collector_meta_path)
    collector_columns = list(collector_meta["feature_columns"])

    def _signature(value):
        return hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    candidate_fit_dir = Path(candidate_runtime["indicator_fit_results_dir"]).as_posix()
    original_fit_dir = Path(source_manifest["indicator_fit_results_dir"]).as_posix()
    volume_same = _signature(feature_config["volume_profile_fixed_range"]) == _signature(
        source_manifest["volume_profile_config"]
    )
    reaction_same = _signature(feature_config["reaction_profile_fixed_grid"]) == _signature(
        source_manifest["reaction_profile_config"]
    )
    comparison = {
        "candidate_feature_count": len(candidate_columns),
        "original_feature_count": len(original_columns),
        "same_ordered_feature_list": candidate_columns == original_columns,
        "new_feature_names": [name for name in candidate_columns if name not in set(original_columns)],
        "removed_feature_names": [name for name in original_columns if name not in set(candidate_columns)],
        "active_general_runtime_model_meta_path": active_meta_path.relative_to(ROOT).as_posix(),
        "active_general_runtime_model_meta_sha256": _sha256(active_meta_path),
        "active_general_runtime_feature_count": len(active_columns),
        "candidate_names_present_in_active_general_runtime": sum(
            name in set(active_columns) for name in candidate_columns
        ),
        "candidate_names_absent_from_active_general_runtime": [
            name for name in candidate_columns if name not in set(active_columns)
        ],
        "active_general_runtime_names_absent_from_candidate_count": sum(
            name not in set(candidate_columns) for name in active_columns
        ),
        "29_feature_preopen_baseline_metadata_path": collector_meta_path.relative_to(ROOT).as_posix(),
        "29_feature_preopen_baseline_metadata_sha256": _sha256(collector_meta_path),
        "29_feature_preopen_baseline_feature_count": len(collector_columns),
        "candidate_names_present_in_29_feature_preopen_baseline": sum(
            name in set(collector_columns) for name in candidate_columns
        ),
        "29_feature_preopen_baseline_scope": "Separate 2026-10-04 causal raw-candle baseline associated with the pre-open collection experiment; not the active 256-column model or this candidate.",
        "indicator_fit_results_dir_candidate": candidate_fit_dir,
        "indicator_fit_results_dir_original": original_fit_dir,
        "indicator_fit_results_dir_same": candidate_fit_dir == original_fit_dir,
        "volume_profile_config_sha256_candidate": _signature(
            feature_config["volume_profile_fixed_range"]
        ),
        "volume_profile_config_sha256_original": _signature(
            source_manifest["volume_profile_config"]
        ),
        "volume_profile_config_same": volume_same,
        "reaction_profile_config_sha256_candidate": _signature(
            feature_config["reaction_profile_fixed_grid"]
        ),
        "reaction_profile_config_sha256_original": _signature(
            source_manifest["reaction_profile_config"]
        ),
        "reaction_profile_config_same": reaction_same,
        "raw_candle_path_original": source_manifest["raw_candle_path"],
        "raw_candle_path_candidate": RAW_PATH.relative_to(ROOT).as_posix(),
        "original_bundle_activated": bool(model_bundle.get("activated", False)),
        "last_tracked_commits_by_source_path": {
            path.relative_to(ROOT).as_posix(): _git_last_commit(path)
            for path in (
                ROOT / "run_btc_preopen_candidate_study.py",
                ROOT / "configs/btc_preopen_v1.json",
                ROOT / "features/candle_features.py",
                ROOT / "features/realized_volatility.py",
                ROOT / "features/volume_profile_fixed_range.py",
                ROOT / "features/reaction_profile_fixed_grid.py",
                ROOT / "features/session_open_features.py",
                ROOT / "features/basis_premium_features.py",
                ROOT / "features/ChaikinOsc.py",
                ROOT / "features/live_indicator_runtime.py",
            )
        },
        "history_limit": (
            "These are the latest tracked commits touching each source path, not claims "
            "that the commit introduced every feature. The generated candidate study "
            "identity does not record a Git commit; its artifact hashes and source paths "
            "are captured separately."
        ),
        "prior_44_of_112_claim_explanation": (
            "The 44 is the exact name intersection of this candidate's 112-column ordered "
            "list and the active BTC model metadata's 256-column list. The remaining 68 "
            "candidate names are absent from that active model bundle, which says nothing "
            "about code support. The separate 29-feature pre-open baseline is another "
            "feature bundle; this audit resolves all 112 candidate columns through the "
            "candidate-configured path."
        ),
        "interpretation": (
            "Candidate booster/calibrator and its runtime wiring differ; the 112 ordered "
            "feature definitions and candidate indicator fit directory and profile configs "
            "match the original model bundle. No feature names, positions, or definitions "
            "were added or changed in this experiment."
        ),
    }
    if not (
            comparison["same_ordered_feature_list"]
            and comparison["indicator_fit_results_dir_same"]
            and volume_same
            and reaction_same
    ):
        raise RuntimeError("Candidate and original feature definitions/configs differ")
    _write_json(REPORT_DIR / "feature_definition_comparison.json", comparison)
    return comparison


class _RssSampler:
    def __init__(self):
        try:
            import psutil

            self.process = psutil.Process()
            self.peak = self.process.memory_info().rss
            self.available = True
        except ImportError:
            self.process = None
            self.peak = None
            self._memory_reader = None
            if os.name == "nt":
                import ctypes
                from ctypes import wintypes

                class _ProcessMemoryCounters(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                kernel32 = ctypes.WinDLL("kernel32")
                psapi = ctypes.WinDLL("psapi")
                process_handle = kernel32.GetCurrentProcess()
                get_memory_info = psapi.GetProcessMemoryInfo
                get_memory_info.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(_ProcessMemoryCounters),
                    wintypes.DWORD,
                ]

                def read_working_set():
                    counters = _ProcessMemoryCounters()
                    counters.cb = ctypes.sizeof(counters)
                    if not get_memory_info(
                        process_handle,
                        ctypes.byref(counters),
                        counters.cb,
                    ):
                        return None
                    return int(counters.WorkingSetSize)

                self._memory_reader = read_working_set
                self.peak = read_working_set()
                self.available = self.peak is not None
            else:
                self.available = False
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if not self.available:
            return
        self.thread = threading.Thread(target=self._run, name="preopen-rss-sampler", daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.wait(0.1):
            try:
                value = (
                    self.process.memory_info().rss
                    if self.process is not None
                    else self._memory_reader()
                )
                if value is not None:
                    self.peak = max(self.peak, value)
            except Exception:
                return

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.available:
            value = (
                self.process.memory_info().rss
                if self.process is not None
                else self._memory_reader()
            )
            if value is not None:
                self.peak = max(self.peak, value)
        return None if self.peak is None else int(self.peak)


def _timing_summary(values_ms):
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "n": int(values.size),
        "p50_ms": float(np.quantile(values, 0.50)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "max_ms": float(np.max(values)),
        "mean_ms": float(np.mean(values)),
    }


def _feature_parity():
    meta = _json(CANDIDATE_META)
    feature_columns = list(meta["feature_columns"])
    feature_config = _json(CANDIDATE_FEATURE_CONFIG)
    history_requirements = _json(CANDIDATE_HISTORY)
    candidate_predictions = pd.read_parquet(
        REPORT_DIR / "candidate_external_predictions.parquet",
        columns=["Opened", "p_candidate_raw", "p_candidate_platt"],
    )
    candidate_predictions["Opened"] = pd.to_datetime(candidate_predictions.Opened, utc=True)
    model = live_runtime.load_model_and_meta(CANDIDATE_META)[0]
    if int(model.num_feature()) != len(feature_columns):
        raise RuntimeError("Candidate LightGBM feature count does not match candidate metadata")

    reference_columns = ["Opened", *live_runtime.RAW_OHLCV_COLS, *feature_columns]
    reference_started = time.perf_counter()
    reference = pd.read_parquet(
        FEATURE_PARQUET,
        columns=reference_columns,
        filters=[
            ("Opened", ">=", PARITY_START.strftime("%Y-%m-%d %H:%M:%S")),
            ("Opened", "<=", PARITY_END.strftime("%Y-%m-%d %H:%M:%S")),
        ],
    )
    reference["Opened"] = pd.to_datetime(reference.Opened, utc=True)
    reference.sort_values("Opened", inplace=True, kind="stable")
    reference.reset_index(drop=True, inplace=True)
    if len(reference) != 422:
        raise RuntimeError(f"Expected 422 reference candles in parity range; found {len(reference)}")
    if reference.Opened.iloc[0] != PARITY_START or reference.Opened.iloc[-1] != PARITY_END:
        raise RuntimeError("Candidate reference parquet does not span the exact parity range")

    rss = _RssSampler()
    rss.start()
    raw_started = time.perf_counter()
    raw = pd.read_csv(
        RAW_PATH,
        usecols=["Opened", *live_runtime.RAW_OHLCV_COLS, "UM_BTCUSDT_Close"],
        parse_dates=["Opened"],
        low_memory=False,
    )
    raw["Opened"] = pd.to_datetime(raw.Opened, utc=True)
    raw.sort_values("Opened", inplace=True, kind="stable")
    raw.reset_index(drop=True, inplace=True)
    raw_read_seconds = time.perf_counter() - raw_started
    if raw.Opened.duplicated().any() or raw.Opened.is_monotonic_increasing is False:
        raise RuntimeError("Candidate raw source has duplicate or out-of-order candle timestamps")

    bootstrap = raw.loc[raw.Opened.le(PARITY_START)].copy()
    if bootstrap.empty or bootstrap.Opened.iloc[-1] != PARITY_START:
        raise RuntimeError("Candidate raw source has no candle at the parity cutoff")
    parity_rows = reference["Opened"].tolist()
    raw_positions = pd.Index(raw.Opened).get_indexer(parity_rows)
    if (raw_positions < 0).any():
        raise RuntimeError("Candidate raw source is missing a reference candle")
    raw_parity = raw.iloc[raw_positions].reset_index(drop=True)
    if not raw_parity["Opened"].equals(reference["Opened"]):
        raise RuntimeError("Candidate raw and feature reference candles are misaligned")
    raw_delta = np.abs(
        raw_parity[list(live_runtime.RAW_OHLCV_COLS)].to_numpy(dtype=np.float64, copy=False)
        - reference[list(live_runtime.RAW_OHLCV_COLS)].to_numpy(dtype=np.float64, copy=False)
    )
    if raw_delta.size and float(np.nanmax(raw_delta)) != 0.0:
        raise RuntimeError("Raw OHLCV inputs differ from the candidate training dataset")
    if not np.array_equal(
        raw_parity["UM_BTCUSDT_Close"].to_numpy(dtype=np.float64, copy=False),
        raw["UM_BTCUSDT_Close"].iloc[raw_positions].to_numpy(dtype=np.float64, copy=False),
    ):
        raise RuntimeError("Candidate futures basis auxiliary input is not stable in raw history")

    construction_started = time.perf_counter()
    retained_candles = int(live_runtime.DEFAULT_BOOTSTRAP_CANDLES)
    profile_vp_state = audit.bootstrap_volume_profile_state_from_history(
        bootstrap.loc[:, ["Opened", "High", "Low", "Volume"]],
        audit.normalize_volume_profile_config(
            feature_config["volume_profile_fixed_range"]
        ),
    )
    profile_rp_state = audit.bootstrap_reaction_profile_state_from_history(
        bootstrap.loc[:, ["Opened", "Open", "High", "Low", "Close"]],
        audit.normalize_reaction_profile_config(
            feature_config["reaction_profile_fixed_grid"]
        ),
    )
    retained_bootstrap = bootstrap.tail(retained_candles).reset_index(drop=True)
    stateful_chaikin_specs = [
        spec
        for spec in live_runtime.load_indicator_specs(
            feature_columns,
            source_label=f"model metadata {CANDIDATE_META}",
            fit_results_dir=INDICATOR_FIT_DIR,
        )
        if spec.indicator == "ChaikinOsc"
        and spec.params.get("fast_ma_type") == "EMA"
        and spec.params.get("slow_ma_type") == "SHMMA"
    ]
    if len(stateful_chaikin_specs) != 1:
        raise RuntimeError(
            "Expected exactly one candidate Chaikin EMA/SHMMA feature to seed."
        )
    chaikin_spec = stateful_chaikin_specs[0]
    chaikin_state_started = time.perf_counter()
    chaikin_state = live_runtime.ChaikinOscillatorRuntimeState.from_history(
        bootstrap.loc[:, live_runtime.OHLCV_COLS].to_numpy(
            dtype=np.float64,
            copy=True,
        ),
        fast_period=chaikin_spec.params["fast_period"],
        slow_period=chaikin_spec.params["slow_period"],
    )
    chaikin_state_rebuild_seconds = time.perf_counter() - chaikin_state_started
    if abs(chaikin_state.value - float(reference[chaikin_spec.feature_col].iloc[0])) > (
        FEATURE_ABS_TOL
        + FEATURE_REL_TOL * abs(float(reference[chaikin_spec.feature_col].iloc[0]))
    ):
        raise RuntimeError(
            "Stateful Chaikin initialization does not reproduce the first training "
            "reference row."
        )
    print(
        f"[candidate-parity] profile bootstrap rows={len(bootstrap):,}; "
        f"retained live candles={len(retained_bootstrap):,} "
        f"range={retained_bootstrap.Opened.iloc[0].isoformat()}->"
        f"{retained_bootstrap.Opened.iloc[-1].isoformat()}",
        flush=True,
    )
    predictor = audit.PseudoLiveAuditPredictor(
        retained_bootstrap,
        model_meta_path=CANDIDATE_META,
        max_keep=retained_candles,
        volume_profile_state=profile_vp_state,
        reaction_profile_state=profile_rp_state,
        feature_runtime_config=feature_config,
        fit_results_dir=INDICATOR_FIT_DIR,
        indicator_history_requirements_path=CANDIDATE_HISTORY,
        indicator_state_by_feature={chaikin_spec.feature_col: chaikin_state},
    )
    predictor.model = model
    initialization_seconds = time.perf_counter() - construction_started
    print(
        f"[candidate-parity] bootstrap complete seconds={initialization_seconds:.2f} "
        f"retained={len(predictor.opened_candles):,} max_keep={predictor.max_keep:,}",
        flush=True,
    )

    live_rows = []
    update_ms, vector_ms, predict_ms, end_to_end_ms = [], [], [], []
    feature_value_mismatch_rows = np.zeros(len(feature_columns), dtype=np.int64)
    feature_mask_mismatch_rows = np.zeros(len(feature_columns), dtype=np.int64)
    feature_max_abs_delta_by_column = np.zeros(len(feature_columns), dtype=np.float64)
    feature_signed_delta_sum = np.zeros(len(feature_columns), dtype=np.float64)
    feature_finite_pair_rows = np.zeros(len(feature_columns), dtype=np.int64)
    resume_seed = None
    resumed_predictor = None
    resumed_diffs = []
    feature_reference_matrix = reference[feature_columns].to_numpy(dtype=np.float64, copy=False)
    reference_raw = model.predict(feature_reference_matrix, num_threads=1)
    reference_raw = np.asarray(reference_raw, dtype=np.float64)
    clipped_reference = np.clip(reference_raw, 1e-6, 1.0 - 1e-6)
    platt = meta["calibration"]
    reference_platt = 1.0 / (1.0 + np.exp(-(
        float(platt["intercept"])
        + float(platt["coefficient"]) * np.log(clipped_reference / (1.0 - clipped_reference))
    )))

    for row_index, row in reference.iterrows():
        opened = pd.Timestamp(row.Opened)
        ohlcv = tuple(float(row[column]) for column in live_runtime.RAW_OHLCV_COLS)
        futures_close = float(raw_parity.loc[row_index, "UM_BTCUSDT_Close"])
        start_ns = time.perf_counter_ns()
        update_start = time.perf_counter_ns()
        if predictor.opened_candles[-1] < opened:
            predictor._append_new_candle(opened, ohlcv, basis_futures_close=futures_close)
        vp = predictor._prepare_volume_profile_features_for_latest_candle(opened)
        rp = predictor._prepare_reaction_profile_features_for_latest_candle(opened)
        update_ms.append((time.perf_counter_ns() - update_start) / 1e6)
        if opened == RESUME_AT:
            resume_seed = {
                "opened": opened,
                "history": pd.DataFrame({
                    "Opened": list(predictor.opened_candles)[-retained_candles:],
                    **{
                        name: predictor.ohlcv_np[-retained_candles:, idx].copy()
                        for idx, name in enumerate(live_runtime.OHLCV_COLS)
                    },
                    predictor.basis_futures_close_col: (
                        predictor.basis_futures_close_np[-retained_candles:].copy()
                    ),
                }),
                "volume_profile_state": copy.deepcopy(predictor.volume_profile_state),
                "reaction_profile_state": copy.deepcopy(predictor.reaction_profile_state),
                "indicator_state_by_feature": copy.deepcopy(
                    predictor.indicator_state_by_feature
                ),
            }
        vector_start = time.perf_counter_ns()
        snapshot = predictor.build_feature_snapshot(
            volume_profile_values=vp,
            reaction_profile_values=rp,
        )
        vector_ms.append((time.perf_counter_ns() - vector_start) / 1e6)
        vector = snapshot["vector"]
        prediction_start = time.perf_counter_ns()
        raw_probability = float(np.asarray(model.predict(vector, num_threads=1)).reshape(-1)[0])
        clipped_probability = float(np.clip(raw_probability, 1e-6, 1.0 - 1e-6))
        calibrated_probability = float(1.0 / (1.0 + np.exp(-(
            float(platt["intercept"])
            + float(platt["coefficient"])
            * np.log(clipped_probability / (1.0 - clipped_probability))
        ))))
        predict_ms.append((time.perf_counter_ns() - prediction_start) / 1e6)
        end_to_end_ms.append((time.perf_counter_ns() - start_ns) / 1e6)

        reference_vector = feature_reference_matrix[row_index]
        finite_live = np.isfinite(vector[0])
        finite_ref = np.isfinite(reference_vector)
        mask_mismatch = int(np.count_nonzero(finite_live != finite_ref))
        allowed = FEATURE_ABS_TOL + FEATURE_REL_TOL * np.abs(reference_vector)
        finite_pair = finite_live & finite_ref
        difference = np.full(len(feature_columns), np.nan, dtype=np.float64)
        difference[finite_pair] = np.abs(vector[0, finite_pair] - reference_vector[finite_pair])
        over_tolerance = finite_pair & (difference > allowed)
        feature_value_mismatch_rows += over_tolerance.astype(np.int64)
        feature_mask_mismatch_rows += (finite_live != finite_ref).astype(np.int64)
        feature_max_abs_delta_by_column = np.maximum(
            feature_max_abs_delta_by_column,
            np.nan_to_num(difference, nan=0.0),
        )
        signed_delta = np.zeros(len(feature_columns), dtype=np.float64)
        signed_delta[finite_pair] = vector[0, finite_pair] - reference_vector[finite_pair]
        feature_signed_delta_sum += signed_delta
        feature_finite_pair_rows += finite_pair.astype(np.int64)
        if np.any(finite_live & ~finite_ref) or np.any(~finite_live & finite_ref):
            feature_error_count = int(np.count_nonzero(over_tolerance))
        else:
            feature_error_count = int(np.count_nonzero(over_tolerance))
        comparable = difference[np.isfinite(difference)]
        max_abs = float(np.max(comparable)) if comparable.size else 0.0
        model_raw_delta = abs(raw_probability - float(reference_raw[row_index]))
        model_platt_delta = abs(calibrated_probability - float(reference_platt[row_index]))
        is_decision = int(opened.minute % 5 == 3)
        live_rows.append({
            "Opened": opened.isoformat(),
            "is_preopen_decision_row": bool(is_decision),
            "is_3m_boundary": opened.minute % 3 == 0,
            "is_5m_boundary": opened.minute % 5 == 0,
            "is_15m_boundary": opened.minute % 15 == 0,
            "is_30m_boundary": opened.minute % 30 == 0,
            "is_1h_boundary": opened.minute == 0,
            "is_4h_boundary": opened.hour % 4 == 0 and opened.minute == 0,
            "is_1d_boundary": opened.hour == 0 and opened.minute == 0,
            "features_compared": len(feature_columns),
            "finite_mask_mismatch_count": mask_mismatch,
            "feature_value_mismatch_count": feature_error_count,
            "feature_max_abs_delta": max_abs,
            "live_raw_probability": raw_probability,
            "reference_raw_probability": float(reference_raw[row_index]),
            "raw_probability_abs_delta": model_raw_delta,
            "live_platt_probability": calibrated_probability,
            "reference_platt_probability": float(reference_platt[row_index]),
            "platt_probability_abs_delta": model_platt_delta,
            "resume_checkpoint_anchor": opened == RESUME_AT,
        })

        if opened == RESUME_AT:
            if resume_seed is None:
                raise RuntimeError("Could not capture candidate profile state for resume check")
            resumed_predictor = audit.PseudoLiveAuditPredictor(
                resume_seed["history"],
                model_meta_path=CANDIDATE_META,
                max_keep=retained_candles,
                volume_profile_state=resume_seed["volume_profile_state"],
                reaction_profile_state=resume_seed["reaction_profile_state"],
                feature_runtime_config=feature_config,
                fit_results_dir=INDICATOR_FIT_DIR,
                indicator_history_requirements_path=CANDIDATE_HISTORY,
                indicator_state_by_feature=resume_seed[
                    "indicator_state_by_feature"
                ],
            )
            resumed_predictor.model = model
        elif resumed_predictor is not None and opened > RESUME_AT:
            resume_ohlcv = tuple(float(row[column]) for column in live_runtime.RAW_OHLCV_COLS)
            resumed_predictor._append_new_candle(
                opened,
                resume_ohlcv,
                basis_futures_close=futures_close,
            )
            resume_vp = resumed_predictor._prepare_volume_profile_features_for_latest_candle(opened)
            resume_rp = resumed_predictor._prepare_reaction_profile_features_for_latest_candle(opened)
            resume_vector = resumed_predictor.build_feature_snapshot(
                volume_profile_values=resume_vp,
                reaction_profile_values=resume_rp,
            )["vector"]
            resumed_delta = np.abs(resume_vector[0] - vector[0])
            finite_resume = np.isfinite(resume_vector[0]) & np.isfinite(vector[0])
            resume_mask_mismatch = int(np.count_nonzero(np.isfinite(resume_vector[0]) != np.isfinite(vector[0])))
            resumed_max_delta = float(np.max(resumed_delta[finite_resume])) if finite_resume.any() else 0.0
            resumed_predict = float(np.asarray(model.predict(resume_vector, num_threads=1)).reshape(-1)[0])
            resumed_diff = abs(resumed_predict - raw_probability)
            resumed_diffs.append({
                "Opened": opened.isoformat(),
                "feature_mask_mismatch_count": resume_mask_mismatch,
                "feature_max_abs_delta": resumed_max_delta,
                "raw_probability_abs_delta": resumed_diff,
            })

    runtime_seed_state = copy.deepcopy(
        predictor.indicator_state_by_feature[chaikin_spec.feature_col]
    )
    raw_tail_after_reference = raw.loc[
        raw.Opened.gt(reference.Opened.iloc[-1]), live_runtime.OHLCV_COLS
    ].to_numpy(dtype=np.float64, copy=False)
    for row in raw_tail_after_reference:
        runtime_seed_state.update(row)
    model_path = Path(meta["artifacts"]["final_model_path"])
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()
    _write_json(
        ROOT / "configs/runtime/btc_preopen_candidate_indicator_state.json",
        {
            "state_version": 1,
            "feature_col": chaikin_spec.feature_col,
            "params": {
                "fast_period": int(chaikin_spec.params["fast_period"]),
                "slow_period": int(chaikin_spec.params["slow_period"]),
            },
            "state_as_of_opened_utc": raw.Opened.iloc[-1].isoformat(),
            "history_start_opened_utc": raw.Opened.iloc[0].isoformat(),
            "raw_source_path": RAW_PATH.relative_to(ROOT).as_posix(),
            "raw_source_sha256": _sha256(RAW_PATH),
            "model_sha256": _sha256(model_path),
            "state": runtime_seed_state.to_dict(),
        },
    )

    feature_diagnostics = pd.read_csv(
        REPORT_DIR / "feature_compatibility_112.csv",
        usecols=["position_1_based", "feature", "family"],
    )
    if feature_diagnostics.feature.tolist() != feature_columns:
        raise RuntimeError("Feature-level diagnostic map does not match candidate feature order")
    feature_diagnostics["value_mismatch_rows"] = feature_value_mismatch_rows
    feature_diagnostics["finite_mask_mismatch_rows"] = feature_mask_mismatch_rows
    feature_diagnostics["max_abs_delta"] = feature_max_abs_delta_by_column
    feature_diagnostics["mean_signed_delta"] = np.divide(
        feature_signed_delta_sum,
        feature_finite_pair_rows,
        out=np.full(len(feature_columns), np.nan, dtype=np.float64),
        where=feature_finite_pair_rows > 0,
    )
    feature_diagnostics.to_csv(REPORT_DIR / "live_feature_parity_by_feature.csv", index=False)

    anchor_frame = pd.DataFrame(live_rows)
    anchor_frame.to_csv(REPORT_DIR / "live_feature_parity_anchors.csv", index=False)
    decision_frame = anchor_frame.loc[anchor_frame.is_preopen_decision_row].copy()
    decision_times = pd.to_datetime(decision_frame.Opened, utc=True)
    expected_predictions = candidate_predictions.loc[
        candidate_predictions.Opened.isin(decision_times)
    ].set_index("Opened")
    decision_frame["Opened_ts"] = decision_times
    decision_frame = decision_frame.join(
        expected_predictions,
        on="Opened_ts",
        how="left",
        rsuffix="_cached",
        validate="one_to_one",
    )
    if decision_frame.p_candidate_raw.isna().any() or decision_frame.p_candidate_platt.isna().any():
        raise RuntimeError("Saved candidate prediction parquet does not cover all pre-open decision anchors")
    cached_raw_delta = np.abs(
        decision_frame.live_raw_probability.to_numpy()
        - decision_frame.p_candidate_raw.to_numpy(dtype=np.float64)
    )
    cached_platt_delta = np.abs(
        decision_frame.live_platt_probability.to_numpy()
        - decision_frame.p_candidate_platt.to_numpy(dtype=np.float64)
    )

    interval_boundaries = {
        "3m": anchor_frame.loc[anchor_frame.is_3m_boundary, "Opened"].iloc[0],
        "5m": anchor_frame.loc[anchor_frame.is_5m_boundary, "Opened"].iloc[0],
        "15m": anchor_frame.loc[anchor_frame.is_15m_boundary, "Opened"].iloc[0],
        "30m": anchor_frame.loc[anchor_frame.is_30m_boundary, "Opened"].iloc[0],
        "1h": anchor_frame.loc[anchor_frame.is_1h_boundary, "Opened"].iloc[0],
        "4h": anchor_frame.loc[anchor_frame.is_4h_boundary, "Opened"].iloc[0],
        "1d": anchor_frame.loc[anchor_frame.is_1d_boundary, "Opened"].iloc[0],
    }
    all_boundary_types_present = all(interval_boundaries.values())
    feature_errors = int(anchor_frame.feature_value_mismatch_count.sum())
    mask_errors = int(anchor_frame.finite_mask_mismatch_count.sum())
    probability_errors = int((anchor_frame.raw_probability_abs_delta > PROBABILITY_ABS_TOL).sum())
    platt_errors = int((anchor_frame.platt_probability_abs_delta > PROBABILITY_ABS_TOL).sum())
    external_raw_errors = int((cached_raw_delta > PROBABILITY_ABS_TOL).sum())
    external_platt_errors = int((cached_platt_delta > PROBABILITY_ABS_TOL).sum())
    resume_failures = sum(
        item["feature_mask_mismatch_count"] > 0
        or item["feature_max_abs_delta"] > FEATURE_ABS_TOL
        or item["raw_probability_abs_delta"] > PROBABILITY_ABS_TOL
        for item in resumed_diffs
    )
    status = "verified" if all((
        feature_errors == 0,
        mask_errors == 0,
        probability_errors == 0,
        platt_errors == 0,
        external_raw_errors == 0,
        external_platt_errors == 0,
        resume_failures == 0,
        all_boundary_types_present,
    )) else "mismatch"
    memory_peak = rss.stop()
    candidate_model_path = Path(
        live_runtime.resolve_model_meta_and_path(CANDIDATE_META)[1]
    )
    if not candidate_model_path.is_absolute():
        candidate_model_path = (ROOT / candidate_model_path).resolve()
    result = {
        "status": status,
        "model_meta_path": CANDIDATE_META.relative_to(ROOT).as_posix(),
        "model_file": candidate_model_path.relative_to(ROOT).as_posix(),
        "feature_count": len(feature_columns),
        "feature_order_sha256_newline_utf8": _sha256_order(feature_columns),
        "history_cutoff_and_span": {
            "bootstrap_start_utc": bootstrap.Opened.iloc[0].isoformat(),
            "first_anchor_inclusive_utc": PARITY_START.isoformat(),
            "last_anchor_inclusive_utc": PARITY_END.isoformat(),
            "bootstrap_rows": int(len(bootstrap)),
            "chronological_rows_compared": int(len(reference)),
            "target_columns_loaded": False,
            "raw_source": RAW_PATH.relative_to(ROOT).as_posix(),
            "reference_dataset": FEATURE_PARQUET.relative_to(ROOT).as_posix(),
        },
        "predeclared_tolerances": {
            "feature_absolute": FEATURE_ABS_TOL,
            "feature_relative": FEATURE_REL_TOL,
            "probability_absolute": PROBABILITY_ABS_TOL,
            "mask_comparison": "finite/non-finite status must match exactly",
        },
        "comparison": {
            "anchor_rows": int(len(anchor_frame)),
            "decision_rows": int(len(decision_frame)),
            "feature_values_compared": int(len(anchor_frame) * len(feature_columns)),
            "feature_value_mismatches": feature_errors,
            "finite_mask_mismatches": mask_errors,
            "features_with_value_mismatches": int(np.count_nonzero(feature_value_mismatch_rows)),
            "features_with_mask_mismatches": int(np.count_nonzero(feature_mask_mismatch_rows)),
            "top_features_by_value_mismatch_rows": feature_diagnostics.sort_values(
                ["value_mismatch_rows", "max_abs_delta"], ascending=False
            ).head(12)[[
                "feature", "family", "value_mismatch_rows",
                "finite_mask_mismatch_rows", "max_abs_delta", "mean_signed_delta",
            ]].to_dict("records"),
            "feature_max_abs_delta": float(anchor_frame.feature_max_abs_delta.max()),
            "raw_probability_max_abs_delta_vs_booster_on_reference": float(anchor_frame.raw_probability_abs_delta.max()),
            "platt_probability_max_abs_delta_vs_booster_on_reference": float(anchor_frame.platt_probability_abs_delta.max()),
            "cached_decision_prediction_raw_max_abs_delta": float(np.max(cached_raw_delta)) if cached_raw_delta.size else None,
            "cached_decision_prediction_platt_max_abs_delta": float(np.max(cached_platt_delta)) if cached_platt_delta.size else None,
            "cached_decision_raw_probability_mismatches": external_raw_errors,
            "cached_decision_platt_probability_mismatches": external_platt_errors,
            "boundary_example_opened_utc": interval_boundaries,
            "resumption_anchor_utc": RESUME_AT.isoformat(),
            "resumption_comparison_rows": len(resumed_diffs),
            "resumption_failures": int(resume_failures),
            "resumption_max_feature_abs_delta": max((row["feature_max_abs_delta"] for row in resumed_diffs), default=0.0),
            "resumption_max_raw_probability_abs_delta": max((row["raw_probability_abs_delta"] for row in resumed_diffs), default=0.0),
        },
        "timing": {
            "raw_csv_read_seconds": float(raw_read_seconds),
            "feature_reference_read_seconds": float(time.perf_counter() - reference_started),
            "candidate_runtime_initialization_seconds": float(initialization_seconds),
            "chaikin_state_rebuild_seconds": float(chaikin_state_rebuild_seconds),
            "warm_state_update": _timing_summary(update_ms),
            "warm_full_feature_vector": _timing_summary(vector_ms),
            "warm_model_predict_plus_platt": _timing_summary(predict_ms),
            "warm_update_vector_predict_end_to_end": _timing_summary(end_to_end_ms),
            "rows_are_local_cpu_only": True,
            "network_latency_measured": False,
            "threads": 1,
        },
        "memory": {
            "rss_peak_sampled_bytes": memory_peak,
            "rss_peak_sampler_interval_ms": 100,
            "sampled_peak_scope": "this audit process; may miss native allocator peaks between samples",
        },
        "retained_history_candles": retained_candles,
        "history_requirement_status": history_requirements.get("validation_status"),
        "profile_states_rebuilt_from_pre_anchor_raw_prefix": True,
    }
    _write_json(REPORT_DIR / "live_feature_parity.json", result)
    if status != "verified":
        raise RuntimeError(f"Candidate live feature parity audit failed: {json.dumps(result['comparison'], indent=2)}")
    return result


def run_audit():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("[compat-audit] replaying corrected BBO freshness for valid direct-book entries", flush=True)
    fresh_bbo = _refresh_quote_validation()
    print("[compat-audit] building all 112 candidate feature mappings", flush=True)
    feature_map = build_feature_map()
    print("[compat-audit] building book rejection flags and daily counts", flush=True)
    coverage, book_summary = _build_book_flags()
    print("[compat-audit] tracing deterministic crossed and BBO-mismatch examples", flush=True)
    traces = _trace_markets(coverage)
    candidate_columns = list(_json(CANDIDATE_META)["feature_columns"])
    original_columns = _json(MODEL_STAGE_DIR / "feature_order.json")["feature_order"]
    runtime_meta = _json(CANDIDATE_META)
    feature_change_audit = _feature_change_audit(candidate_columns, original_columns)
    print("[compat-audit] replaying candidate live feature path on local historical candles", flush=True)
    parity = _feature_parity()
    manifest = _artifact_manifest(candidate_columns, original_columns)
    result = {
        "report_version": 1,
        "book_diagnostics": book_summary,
        "fresh_bbo_validation": {
            "target_markets": fresh_bbo["target_market_count"],
            "target_entries": fresh_bbo["target_entry_count"],
            "partitions_scanned": fresh_bbo["partitions_scanned"],
            "rows_applied_to_target_market_states": fresh_bbo[
                "rows_applied_to_target_market_states"
            ],
            "elapsed_seconds": fresh_bbo["elapsed_seconds"],
            "peak_sampled_rss_bytes": fresh_bbo["peak_sampled_rss_bytes"],
            "eligible_before": fresh_bbo["eligible_before_by_entry_case"],
            "eligible_after": fresh_bbo["eligible_after_by_entry_case"],
            "mismatches_cleared_as_stale_or_tied": fresh_bbo[
                "ask_mismatches_rejected_as_old_or_tied_references"
            ],
            "mismatches_still_confirmed": fresh_bbo[
                "ask_mismatches_still_confirmed_by_fresh_reference"
            ],
        },
        "feature_map_rows": len(feature_map),
        "feature_family_counts": pd.Series([row["family"] for row in feature_map]).value_counts().to_dict(),
        "feature_order_exactly_matches_original": candidate_columns == original_columns,
        "candidate_feature_definition_audit": feature_change_audit,
        "feature_definition_comparison_path": "reports/btc_preopen/feature_definition_comparison.json",
        "candidate_runtime_manifest": str(CANDIDATE_RUNTIME.relative_to(ROOT)).replace("\\", "/"),
        "candidate_model_meta_feature_count": len(runtime_meta["feature_columns"]),
        "book_trace_examples": [
            {
                "purpose": item["trace_purpose"],
                "condition_id": item["condition_id"],
                "market_slug": item["market_slug"],
            }
            for item in traces
        ],
        "artifact_manifest_path": "reports/btc_preopen/artifact_manifest.json",
        "archive_partition_hashes_path": "reports/btc_preopen/archive_partition_hashes.csv",
        "live_feature_parity_artifact_path": "reports/btc_preopen/live_feature_parity.json",
        "pmxt_partition_schema": {
            "partition_count": manifest["pmxt_cache"]["partition_count"],
            "observed_schema_count": manifest["pmxt_cache"]["observed_partition_schema_count"],
            "schema_version_column_present": manifest["pmxt_cache"]["schema_version_column_present"],
            "schema_sequence_id_present": manifest["pmxt_cache"]["schema_sequence_id_present"],
            "schemas": manifest["pmxt_cache"]["observed_partition_schemas"],
        },
        "archive_event_type_counts": manifest["pmxt_cache"]["event_type_counts"],
        "feature_parity": {
            "status": parity["status"],
            "anchors": parity["comparison"]["anchor_rows"],
            "decision_rows": parity["comparison"]["decision_rows"],
            "feature_mismatches": parity["comparison"]["feature_value_mismatches"],
            "mask_mismatches": parity["comparison"]["finite_mask_mismatches"],
            "resume_mismatches": parity["comparison"]["resumption_failures"],
            "comparison": parity["comparison"],
            "predeclared_tolerances": parity["predeclared_tolerances"],
            "timing": parity["timing"],
            "memory": parity["memory"],
        },
        "candidate_and_original_model_feature_order_hashes": {
            "candidate_newline_sha256": manifest["candidate"]["ordered_feature_sha256_newline_utf8"],
            "original_newline_sha256": manifest["original_v1"]["ordered_feature_sha256_newline_utf8"],
        },
        "historical_pnl_interpretation": "The cached result is retrospective and historically exposed, not an independent holdout. The corrected locked-quote validation changes eligibility; economics were recomputed from saved entry snapshots only, without rerunning PMXT book reconstruction.",
    }
    _write_json(REPORT_DIR / "runtime_compatibility_audit.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_audit()
