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
ENTRY_SNAPSHOTS = REPORT_DIR / "entry_snapshots.parquet"
BOOK_TIMING_EXAMPLES = REPORT_DIR / "book_timing_examples.csv"
ECONOMIC_COMPARISON = REPORT_DIR / "primary_economic_comparison.csv"
QUOTE_IMPACT = REPORT_DIR / "quote_validation_economic_impact.csv"
QUOTE_MARKET_CHANGES = REPORT_DIR / "quote_validation_market_changes.csv"
CORRECTED_TRADES = REPORT_DIR / "quote_validation_corrected_trades.parquet"
ENTRY_SNAPSHOT_SIMULATION_COLUMNS = [
    "entry_case", "entry_time_utc", "prediction_available_at_utc",
    "condition_id", "entry_kind", "market_slug", "market_start_utc",
    "compute_delay_seconds", "order_delay_seconds", "target_polymarket_up",
    "resolved_at_utc", "fee_collection_mode", "no_future_event_at_entry",
    "no_future_source_event_at_entry", "has_full_snapshot", "quote_valid",
    "bbo_at_entry_ask_mismatches", "up_best_ask", "down_best_ask",
    "up_best_ask_size_shares", "down_best_ask_size_shares",
    "up_quote_source_token_id", "down_quote_source_token_id",
    "up_quote_complemented", "down_quote_complemented",
    "up_ask_age_seconds", "down_ask_age_seconds", "fee_rate_bps",
    "fee_known", "depth_5usd_valid_both_sides", "up_fill", "down_fill",
    "p_model_raw", "p_model_platt", "p_candidate_raw", "p_candidate_platt",
]


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
    if pd.isna(value):
        return "-"
    amount = float(value)
    return f"-${abs(amount):.2f}" if amount < 0.0 else f"${amount:.2f}"


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

def _build_primary_economic_comparison(coverage):
    import run_btc_preopen_economic_replay as replay

    snapshots = pd.read_parquet(
        ENTRY_SNAPSHOTS,
        columns=ENTRY_SNAPSHOT_SIMULATION_COLUMNS,
    )
    primary = snapshots.loc[snapshots.entry_case.eq("prestart_c0_o1")].copy()
    primary["entry_time_utc"] = pd.to_datetime(primary.entry_time_utc, utc=True)
    eligible_ids = set(
        coverage.loc[
            coverage.entry_case.eq("prestart_c0_o1")
            & coverage.coverage_reason_age30s.eq("eligible"),
            "condition_id",
        ]
    )
    common = primary.loc[primary.condition_id.isin(eligible_ids)].copy()
    if common.condition_id.nunique() != len(eligible_ids) or len(common) != len(eligible_ids):
        raise RuntimeError("Cached entry snapshots do not cover the age-30 eligible markets")

    model_names = (
        "candidate_platt",
        "original_v1_platt",
        "candidate_raw",
        "original_v1_raw",
        "no_btc_constant_0_5",
        "no_btc_development_prevalence_0_5015364895",
    )
    baseline_probabilities = {
        "no_btc_constant_0_5": 0.5,
        "no_btc_development_prevalence_0_5015364895": 0.5015364895,
    }
    rows = []
    for sample_scope, group in (
        ("all_archived_markets", primary),
        ("shared_eligible_markets", common),
    ):
        for model_name in model_names:
            simulation_group = group
            simulation_name = model_name
            if model_name in baseline_probabilities:
                simulation_group = group.copy()
                simulation_group["p_candidate_platt"] = baseline_probabilities[model_name]
                simulation_name = "candidate_platt"
            result, _ = replay._simulate_group(
                simulation_group,
                simulation_name,
                max_age_seconds=30,
                release_delay_seconds=60,
                keep_trades=False,
            )
            result["model"] = model_name
            rows.append({
                "sample_scope": sample_scope,
                "sample_markets": int(group.condition_id.nunique()),
                "model": model_name,
                "baseline_probability": baseline_probabilities.get(model_name),
                "entry_case": "prestart_c0_o1",
                "max_ask_age_seconds": 30,
                "settlement_release_delay_seconds": 60,
                "initial_cash_usd": result["initial_cash_usd"],
                "ending_cash_usd": result["ending_cash_usd"],
                "net_pnl_usd": result["net_pnl_usd"],
                "max_drawdown_at_cost": result["max_drawdown_at_cost"],
                "trade_count": result["trade_count"],
                "gross_turnover_usd": result["gross_turnover_usd"],
                "fees_paid_usd": result["fees_paid_usd"],
                "data_rejections": result.get("data_rejections", 0),
                "skip_no_positive_expected_edge": result.get("skip_no_positive_expected_edge", 0),
                "reject_insufficient_balance": result.get("reject_insufficient_balance", 0),
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(ECONOMIC_COMPARISON, index=False)
    return frame


def _build_quote_validation_economic_impact(coverage):
    import run_btc_preopen_economic_replay as replay

    snapshot_columns = list(dict.fromkeys(
        ENTRY_SNAPSHOT_SIMULATION_COLUMNS
        + ["quote_valid_strict_bid_lt_ask"]
    ))
    snapshots = pd.read_parquet(ENTRY_SNAPSHOTS, columns=snapshot_columns)
    snapshots = snapshots.loc[snapshots.entry_case.eq("prestart_c0_o1")].copy()
    snapshots["entry_time_utc"] = pd.to_datetime(snapshots.entry_time_utc, utc=True)
    snapshots["condition_id"] = snapshots.condition_id.astype(str)

    before = pd.read_csv(REPORT_DIR / "quote_validation_before_fresh_bbo.csv")
    before = before.loc[before.entry_case.eq("prestart_c0_o1")].copy()
    before["condition_id"] = before.condition_id.astype(str)
    before = before[[
        "condition_id", "quote_valid", "quote_valid_strict_bid_lt_ask",
        "bbo_at_entry_ask_mismatches", "coverage_reason_age30s",
    ]].rename(columns={
        "quote_valid": "quote_valid_before_fresh_bbo",
        "quote_valid_strict_bid_lt_ask": "quote_valid_strict_before_fresh_bbo",
        "bbo_at_entry_ask_mismatches": "ask_mismatches_before_fresh_bbo",
        "coverage_reason_age30s": "reason_before_fresh_bbo",
    })
    after = coverage.loc[coverage.entry_case.eq("prestart_c0_o1"), [
        "condition_id", "coverage_reason_age30s",
    ]].copy()
    after["condition_id"] = after.condition_id.astype(str)
    data = snapshots.merge(before, on="condition_id", how="left", validate="one_to_one")
    data = data.merge(after, on="condition_id", how="left", validate="one_to_one")
    if data.quote_valid_before_fresh_bbo.isna().any() or data.coverage_reason_age30s.isna().any():
        raise RuntimeError("T-59 snapshots are missing before/after quote-validation rows")

    stages = {
        "original_strict_bid_lt_ask": {
            "quote_valid": (
                data.quote_valid_before_fresh_bbo.fillna(False).astype(bool)
                & data.quote_valid_strict_before_fresh_bbo.fillna(False).astype(bool)
            ),
            "ask_mismatches": data.ask_mismatches_before_fresh_bbo,
        },
        "locked_quotes_old_bbo_freshness": {
            "quote_valid": data.quote_valid_before_fresh_bbo.fillna(False).astype(bool),
            "ask_mismatches": data.ask_mismatches_before_fresh_bbo,
        },
        "locked_quotes_corrected_bbo_freshness": {
            "quote_valid": data.quote_valid.fillna(False).astype(bool),
            "ask_mismatches": data.bbo_at_entry_ask_mismatches,
        },
    }
    stage_frames = {}
    stage_eligible_ids = {}
    for stage, config in stages.items():
        frame = data.copy()
        frame["quote_valid"] = config["quote_valid"].to_numpy()
        frame["bbo_at_entry_ask_mismatches"] = pd.to_numeric(
            config["ask_mismatches"], errors="coerce"
        ).fillna(0).to_numpy()
        frame["quote_validation_reason"] = frame.apply(
            lambda row: replay._data_reason(row, 30), axis=1
        )
        stage_frames[stage] = frame
        stage_eligible_ids[stage] = set(
            frame.loc[frame.quote_validation_reason.eq("eligible"), "condition_id"]
        )

    before_artifact = _json(REPORT_DIR / "book_validation_before.json")
    expected_original_eligible = int(
        before_artifact["reason_counts_age30_seconds"]["prestart_c0_o1"]["eligible"]
    )
    if len(stage_eligible_ids["original_strict_bid_lt_ask"]) != expected_original_eligible:
        raise RuntimeError("Reconstructed original strict-quote eligibility differs from saved before-fix audit")
    expected_fresh_eligible = int(
        coverage.loc[
            coverage.entry_case.eq("prestart_c0_o1"),
            "coverage_reason_age30s",
        ].eq("eligible").sum()
    )
    if len(stage_eligible_ids["locked_quotes_corrected_bbo_freshness"]) != expected_fresh_eligible:
        raise RuntimeError("Reconstructed corrected quote eligibility differs from final coverage")

    common_ids = set.intersection(*stage_eligible_ids.values())
    model_names = (
        "candidate_platt", "original_v1_platt", "candidate_raw", "original_v1_raw",
        "no_btc_constant_0_5", "no_btc_development_prevalence_0_5015364895",
    )
    baseline_probabilities = {
        "no_btc_constant_0_5": 0.5,
        "no_btc_development_prevalence_0_5015364895": 0.5015364895,
    }
    impact_rows = []
    for stage, frame in stage_frames.items():
        scopes = (
            ("all_archived_markets", frame),
            ("stage_eligible_markets", frame.loc[
                frame.condition_id.isin(stage_eligible_ids[stage])
            ].copy()),
            ("common_eligible_across_stages", frame.loc[
                frame.condition_id.isin(common_ids)
            ].copy()),
        )
        for scope, group in scopes:
            for model_name in model_names:
                simulation_group = group
                simulation_name = model_name
                if model_name in baseline_probabilities:
                    simulation_group = group.copy()
                    simulation_group["p_candidate_platt"] = baseline_probabilities[model_name]
                    simulation_name = "candidate_platt"
                result, _ = replay._simulate_group(
                    simulation_group,
                    simulation_name,
                    max_age_seconds=30,
                    release_delay_seconds=60,
                    keep_trades=False,
                )
                impact_rows.append({
                    "validation_stage": stage,
                    "sample_scope": scope,
                    "sample_markets": int(group.condition_id.nunique()),
                    "eligible_markets_at_stage": len(stage_eligible_ids[stage]),
                    "model": model_name,
                    "baseline_probability": baseline_probabilities.get(model_name),
                    "net_pnl_usd": result["net_pnl_usd"],
                    "ending_cash_usd": result["ending_cash_usd"],
                    "max_drawdown_at_cost": result["max_drawdown_at_cost"],
                    "trade_count": result["trade_count"],
                    "gross_turnover_usd": result["gross_turnover_usd"],
                    "fees_paid_usd": result["fees_paid_usd"],
                    "data_rejections": result.get("data_rejections", 0),
                    "skip_no_positive_expected_edge": result.get("skip_no_positive_expected_edge", 0),
                    "reject_insufficient_balance": result.get("reject_insufficient_balance", 0),
                })

    impact = pd.DataFrame(impact_rows)
    common = impact.loc[impact.sample_scope.eq("common_eligible_across_stages")]
    for model_name, group in common.groupby("model"):
        if group.net_pnl_usd.max() - group.net_pnl_usd.min() > 1e-9:
            raise RuntimeError(f"Quote validation changed economics on common eligible markets for {model_name}")
    saved_original = {
        (row["sample_scope"], row["model"]): row
        for row in before_artifact["primary_t59_economics"]
    }
    reconstructed_original = impact.loc[
        impact.validation_stage.eq("original_strict_bid_lt_ask")
        & impact.sample_scope.isin(("all_archived_markets", "stage_eligible_markets"))
    ]
    for row in reconstructed_original.itertuples(index=False):
        saved_scope = (
            "all_archived_markets"
            if row.sample_scope == "all_archived_markets"
            else "shared_eligible_markets"
        )
        expected = saved_original[(saved_scope, row.model)]
        for metric in ("net_pnl_usd", "ending_cash_usd", "max_drawdown_at_cost"):
            if abs(float(getattr(row, metric)) - float(expected[metric])) > 1e-8:
                raise RuntimeError(
                    f"Reconstructed original strict-quote {metric} differs for {row.sample_scope}/{row.model}"
                )
        if row.trade_count != expected["trade_count"]:
            raise RuntimeError(
                f"Reconstructed original strict-quote trade count differs for {row.sample_scope}/{row.model}"
            )
    impact.to_csv(QUOTE_IMPACT, index=False)

    corrected_trade_rows = []
    corrected_frame = stage_frames["locked_quotes_corrected_bbo_freshness"]
    for model_name in model_names:
        simulation_group = corrected_frame
        simulation_name = model_name
        if model_name in baseline_probabilities:
            simulation_group = corrected_frame.copy()
            simulation_group["p_candidate_platt"] = baseline_probabilities[model_name]
            simulation_name = "candidate_platt"
        _, trade_rows = replay._simulate_group(
            simulation_group,
            simulation_name,
            max_age_seconds=30,
            release_delay_seconds=60,
            keep_trades=True,
        )
        for trade in trade_rows:
            trade["model"] = model_name
            trade["validation_stage"] = "locked_quotes_corrected_bbo_freshness"
        corrected_trade_rows.extend(trade_rows)
    corrected_trades = pd.DataFrame(corrected_trade_rows)
    corrected_trades.to_parquet(CORRECTED_TRADES, index=False)

    market_changes = data[[
        "condition_id", "market_slug", "market_start_utc", "entry_time_utc",
        "reason_before_fresh_bbo", "coverage_reason_age30s",
        "quote_valid_strict_before_fresh_bbo", "quote_valid_before_fresh_bbo",
        "quote_valid", "ask_mismatches_before_fresh_bbo", "bbo_at_entry_ask_mismatches",
    ]].rename(columns={
        "coverage_reason_age30s": "reason_after_fresh_bbo",
        "quote_valid": "quote_valid_after_fresh_bbo",
        "bbo_at_entry_ask_mismatches": "ask_mismatches_after_fresh_bbo",
    })
    market_changes["reason_original_strict_bid_lt_ask"] = stage_frames[
        "original_strict_bid_lt_ask"
    ].quote_validation_reason.to_numpy()
    market_changes["reason_locked_quotes_old_bbo_freshness"] = stage_frames[
        "locked_quotes_old_bbo_freshness"
    ].quote_validation_reason.to_numpy()
    for stage, ids in stage_eligible_ids.items():
        market_changes[f"eligible_{stage}"] = market_changes.condition_id.isin(ids)
    original_ids = stage_eligible_ids["original_strict_bid_lt_ask"]
    locked_ids = stage_eligible_ids["locked_quotes_old_bbo_freshness"]
    fresh_ids = stage_eligible_ids["locked_quotes_corrected_bbo_freshness"]
    market_changes["validation_transition"] = "unchanged"
    market_changes.loc[
        ~market_changes.condition_id.isin(original_ids)
        & market_changes.condition_id.isin(locked_ids),
        "validation_transition",
    ] = "added_by_accepting_locked_quote"
    market_changes.loc[
        market_changes.condition_id.isin(locked_ids)
        & ~market_changes.condition_id.isin(fresh_ids),
        "validation_transition",
    ] = "removed_by_fresh_bbo_reconciliation"
    market_changes.loc[
        ~market_changes.condition_id.isin(locked_ids)
        & market_changes.condition_id.isin(fresh_ids),
        "validation_transition",
    ] = "added_by_fresh_bbo_reconciliation"
    market_changes.to_csv(QUOTE_MARKET_CHANGES, index=False)
    return impact, market_changes, corrected_trades


def _build_completed_summary(
        *,
        examples,
        market_index,
        coverage,
        comparison,
        quote_impact,
        quote_changes,
        compatibility,
        trades,
        live_timing,
        training,
):
    index_by_id = market_index.set_index("condition_id")
    example_lines = []
    for row in examples.itertuples(index=False):
        market = index_by_id.loc[row.condition_id]
        example_lines.append(
            f"| {row.period_position} | `{row.condition_id}` / `{row.market_slug}` | "
            f"{row.market_start_utc} | {row.entry_time_utc} | "
            f"{row.first_observed_archive_event_received_utc} | "
            f"{row.up_initial_full_book_received_utc} / {row.down_initial_full_book_received_utc} | "
            f"{row.last_any_archive_event_received_utc} | "
            f"{row.up_last_changed_ask_depth_received_utc} ({row.up_last_changed_ask_depth_source_utc}; age {row.up_ask_age_seconds:.3f}s) / "
            f"{row.down_last_changed_ask_depth_received_utc} ({row.down_last_changed_ask_depth_source_utc}; age {row.down_ask_age_seconds:.3f}s) | "
            f"UP `{market.up_token_id}`: {row.up_best_bid:.2f}/{row.up_best_ask:.2f}, "
            f"{row.up_best_ask_size_shares:.2f} shares, ${row.up_5usd_ask_vwap:.2f}; "
            f"DOWN `{market.down_token_id}`: {row.down_best_bid:.2f}/{row.down_best_ask:.2f}, "
            f"{row.down_best_ask_size_shares:.2f} shares, ${row.down_5usd_ask_vwap:.2f} |"
        )

    t60 = coverage.loc[coverage.entry_case.eq("prestart_c0_o0")]
    t59 = coverage.loc[coverage.entry_case.eq("prestart_c0_o1")]
    t60_both_books = int(t60.has_full_snapshot.sum())
    t60_both_depth = int(t60.depth_5usd_valid_both_sides.sum())
    t60_valid_bbo = int(t60.quote_valid.sum())
    t59_eligible = int(t59.coverage_reason_age30s.eq("eligible").sum())
    t59_reason_counts = {
        str(reason): int(count)
        for reason, count in t59.coverage_reason_age30s.value_counts().items()
    }
    old_t59_reason_counts = _json(REPORT_DIR / "book_validation_before.json")[
        "reason_counts_age30_seconds"
    ]["prestart_c0_o1"]
    pre_fresh_bbo_t59_reasons = _json(REPORT_DIR / "fresh_bbo_replay_summary.json")[
        "reason_counts_before_by_entry_case"
    ]["prestart_c0_o1"]
    schema_audit = compatibility["pmxt_partition_schema"]
    event_type_counts = compatibility["archive_event_type_counts"]
    fresh_bbo_audit = compatibility["fresh_bbo_validation"]
    feature_parity = compatibility["feature_parity"]
    feature_definition_audit = compatibility["candidate_feature_definition_audit"]
    feature_timing = feature_parity["timing"]
    feature_memory = feature_parity["memory"]
    feature_families = compatibility["feature_family_counts"]
    state_seed_as_of = _json(
        ROOT / "configs/runtime/btc_preopen_candidate_indicator_state.json"
    )["state_as_of_opened_utc"]
    bbo_time_relations = compatibility["book_diagnostics"]["prestart_c0_o1"][
        "entry_reference_bbo_source_time_relations"
    ]
    t60_silence = (
        pd.to_datetime(t60.entry_time_utc, utc=True, format="mixed")
        - pd.to_datetime(t60.market_last_event_utc, utc=True, format="mixed")
    ).dt.total_seconds().dropna()
    silence_stats = {
        "p50": float(t60_silence.quantile(0.50)),
        "p95": float(t60_silence.quantile(0.95)),
        "p99": float(t60_silence.quantile(0.99)),
        "max": float(t60_silence.max()),
    }

    comparison_lines = []
    for row in comparison.itertuples(index=False):
        scope = "all archived" if row.sample_scope == "all_archived_markets" else "common eligible"
        comparison_lines.append(
            f"| {scope} ({row.sample_markets:,}) | `{row.model}` | "
            f"{_money(row.net_pnl_usd)} | {_money(row.ending_cash_usd)} | "
            f"{float(row.max_drawdown_at_cost):.2%} | {row.trade_count:,} | "
            f"{_money(row.gross_turnover_usd)} | {_money(row.fees_paid_usd)} | "
            f"{row.data_rejections:,} / {row.skip_no_positive_expected_edge:,} / "
            f"{row.reject_insufficient_balance:,} |"
        )

    quote_impact_lines = []
    eligible_impact = quote_impact.loc[
        quote_impact.sample_scope.eq("stage_eligible_markets")
    ]
    for row in eligible_impact.itertuples(index=False):
        quote_impact_lines.append(
            f"| `{row.validation_stage}` ({row.eligible_markets_at_stage:,}) | `{row.model}` | "
            f"{_money(row.net_pnl_usd)} | {row.trade_count:,} |"
        )
    quote_transition_counts = quote_changes.validation_transition.value_counts().to_dict()
    candidate_common_quote_result = quote_impact.loc[
        quote_impact.validation_stage.eq("original_strict_bid_lt_ask")
        & quote_impact.sample_scope.eq("common_eligible_across_stages")
        & quote_impact.model.eq("candidate_platt")
    ].iloc[0]
    candidate_corrected_quote_result = quote_impact.loc[
        quote_impact.validation_stage.eq("locked_quotes_corrected_bbo_freshness")
        & quote_impact.sample_scope.eq("stage_eligible_markets")
        & quote_impact.model.eq("candidate_platt")
    ].iloc[0]

    main_trades = trades.loc[
        trades.entry_case.eq("prestart_c0_o1")
        & trades.max_ask_age_seconds.eq(30)
        & trades.settlement_release_delay_seconds.eq(60)
        & trades.model.eq("candidate_platt")
    ]
    expected_candidate_trades = int(comparison.loc[
        comparison.model.eq("candidate_platt")
        & comparison.sample_scope.eq("all_archived_markets"),
        "trade_count",
    ].iloc[0])
    if len(main_trades) != expected_candidate_trades:
        raise RuntimeError("Corrected candidate trade log does not match the T-59 economics table")
    cash_after = pd.to_numeric(main_trades.cash_available_after_entry_usd, errors="coerce")
    cash_before = pd.to_numeric(main_trades.cash_available_before_usd, errors="coerce")
    debit = pd.to_numeric(main_trades.cash_debit_usd, errors="coerce")
    negative_cash_count = int((cash_after < -1e-9).sum())
    debit_violation_count = int((cash_before + 1e-9 < debit).sum())
    minimum_cash_after = float(cash_after.min()) if cash_after.notna().any() else float("nan")

    historical = {item["stage"]: item for item in live_timing["historical_live_stages"]}
    old_cycle = historical["cycle_complete_from_window_start_ms"]
    old_submit = historical["submit_call_including_response_ms"]
    collection = live_timing["preopen_collection"]
    input_offsets = collection["all_execution_inputs_ready"]["observed_offsets_ms"]
    prevalence = comparison.loc[
        comparison.model.eq("no_btc_development_prevalence_0_5015364895")
        & comparison.sample_scope.eq("shared_eligible_markets")
    ].iloc[0]
    constant_baseline = comparison.loc[
        comparison.model.eq("no_btc_constant_0_5")
        & comparison.sample_scope.eq("shared_eligible_markets")
    ].iloc[0]
    candidate_platt = comparison.loc[
        comparison.model.eq("candidate_platt")
        & comparison.sample_scope.eq("shared_eligible_markets")
    ].iloc[0]
    original_platt = comparison.loc[
        comparison.model.eq("original_v1_platt")
        & comparison.sample_scope.eq("shared_eligible_markets")
    ].iloc[0]

    return [
        "# BTC pre-open: ocena eksperymentu i telemetria live",
        "",
        "Główny scenariusz ekonomiczny to ustalone wejście T−59 s: sekundę po nominalnej decyzji T−60 s. To założenie operacyjne dla przyszłego uruchomienia serwera. Nie jest zmierzonym maksimum ani gwarantowanym worst case. Warianty wcześniejszych analiz pozostają w `economic_scenarios.csv`; dalsze porównania w tym raporcie dotyczą T−59 s.",
        "",
        "Nie pobierano ponownie archiwum. Ukierunkowany replay lokalnych partycji PMXT odtworzył historię dla kompletnych, semantycznie poprawnych booków T−60/T−59, aby zweryfikować świeżość referencji BBO; zakres i zasoby są zapisane w raporcie. Ekonomikę policzono ponownie z istniejących snapshotów i filli, bo korekta zmienia kwalifikację rynków. Nie wysłano prawdziwych zleceń i nie uruchomiono handlu live.",
        "",
        f"Weryfikacja BBO objęła {fresh_bbo_audit['target_markets']:,} rynków / {fresh_bbo_audit['target_entries']:,} snapshotów, odczytała {fresh_bbo_audit['partitions_scanned']:,} lokalnych partycji, zastosowała {fresh_bbo_audit['rows_applied_to_target_market_states']:,} zdarzeń do stanów docelowych i trwała {fresh_bbo_audit['elapsed_seconds'] / 60:.1f} min. Szczyt RSS próbkowany co 100 ms wyniósł {fresh_bbo_audit['peak_sampled_rss_bytes'] / (1024**2):.0f} MiB; nie było ruchu sieciowego ani pobierania danych.",
        "",
        "## Czas i pochodzenie ceny wejścia",
        "",
        "Tak: archiwum mapuje każde `condition_id` do natywnych tokenów UP/DOWN przez indeks rynku i zapisane mapowanie tokenów. Oficjalny start T pochodzi z indeksu rynku / bucketa sluga; dla przykładowych rynków poniżej slug epoch zgadza się z T. Replay używa aktualizacji według `timestamp_received` kolektora archiwum. Snapshot T−59 obejmuje zdarzenia odebrane do tej chwili włącznie, w tym zmiany rozmiaru i usunięcia poziomów; później odebrane zdarzenia są wykluczone nawet wtedy, gdy ich czas źródłowy wygląda na wcześniejszy. Zdarzenia z czasem źródłowym po wejściu są także odrzucane przez kontrolę przyczynowości. PMXT nie podaje monotonicznego identyfikatora kolejności: zdarzenia z identycznymi receive/source timestamp mają tylko stabilny porządek w części Parquet, nie gwarantowaną kolejność giełdową. Referencję BBO uznajemy za rozstrzygającą tylko, gdy jej czas źródłowy jest późniejszy od ostatniej zmiany obu stron. Starsza lub równa referencja jest raportowana osobno, bo bez identyfikatora sekwencji nie ustala kolejności; świeża rozbieżność ask wyklucza snapshot.",
        "",
        "Pełny `book` jest inicjalizatorem stanu, nie ceną zakupu. Po nim replay składa stan z wcześniejszych zmian poziomów. Fill $5 przechodzi po natywnych poziomach ask właściwego tokena, uwzględniając dostępną głębokość i opłaty; komplementowane kwotowanie nie dostarcza głębokości do fillu. Zapisany replay raportuje 0 snapshotów skażonych zdarzeniami odebranymi po wejściu i 0 snapshotów ze zdarzeniem źródłowym po wejściu.",
        "",
        f"Przy T−60 oba natywne booki były zainicjalizowane dla {t60_both_books:,}/{len(t60):,} rynków; dla {t60_valid_bbo:,} BBO obu stron były poprawne, a dla {t60_both_depth:,} obie strony miały głębokość wystarczającą na $5. Przy T−59, filtrze ask age 30 s i pozostałych warunkach kwalifikuje się {t59_eligible:,}/{len(t59):,} rynków; powody z cache: `{json.dumps(t59_reason_counts, ensure_ascii=False)}`.",
        f"The former 75% rejection rate was {100.0 * (len(t59) - old_t59_reason_counts['eligible']) / len(t59):.1f}% ({len(t59) - old_t59_reason_counts['eligible']:,}/{len(t59):,}), not proof that those markets had no exchange liquidity. The earlier strict `bid < ask` rule rejected valid locked books; after accepting locks, {fresh_bbo_audit['eligible_before']['prestart_c0_o1']:,} qualified (+{quote_transition_counts.get('added_by_accepting_locked_quote', 0):,} vs the old strict stage). The old-BBO stage then rejected {pre_fresh_bbo_t59_reasons['unreconciled_best_ask_at_entry']:,} ask mismatches. The timestamp gate counted {fresh_bbo_audit['mismatches_cleared_as_stale_or_tied']:,} prior BBO disagreements cleared as older/tied across T-60 and T-59; at T-59 only {t59_reason_counts.get('unreconciled_best_ask_at_entry', 0):,} fresh ask mismatches remain, and {t59_eligible:,} qualify (+{quote_transition_counts.get('added_by_fresh_bbo_reconciliation', 0):,} net). Current exclusions are {len(t59) - t59_eligible:,}, including {t59_reason_counts.get('incomplete_or_crossed_book', 0):,} incomplete/crossed books. The full primary-reason sums and overlapping flags are in the audit tables; none of these counts establish exchange liquidity where the archive is incomplete.",
        "",
        f"`ask age` to wiek ostatniej rzeczywistej zmiany dowolnego poziomu ask w natywnej księdze: dodanie, zmiana ceny/rozmiaru albo usunięcie poziomu odświeża go, także poza best ask; identyczny duplikat nie. Pełny snapshot inicjalizuje oba booki i resetuje wiek. Zmiana rozmiaru ≤0 usuwa poziom. Osobne `ask_received_age` używa czasu odbioru archiwizatora, a filtr 30 s korzysta z czasu źródłowego (zastępowanego czasem odbioru, jeśli źródła brak). To wiek zmienionej głębokości, nie opóźnienie wejścia ani miara ciągłości feedu.",
        "",
        f"Przerwa między ostatnią wiadomością odebraną przez archiwizator a wejściem T−60 miała p50 {silence_stats['p50']:.3f} s, p95 {silence_stats['p95']:.3f} s, p99 {silence_stats['p99']:.3f} s i maksimum {silence_stats['max']:.3f} s. To cisza w archiwalnym odbiorze, nie dowód braku zdarzeń na giełdzie ani jakość feedu hipotetycznego serwera. Pierwsza obserwacja archiwalna oznacza pierwsze zdarzenie zobaczone przez eksportera, nie moment publikacji rynku przez giełdę.",
        "",
        f"Kontrola schematu wykazała {schema_audit['partition_count']:,} lokalnych partycji PMXT i {schema_audit['observed_schema_count']} wariantów kolumn; pole `schema_version` i monotoniczny event sequence ID nie występują. Liczniki archiwum to `{json.dumps(event_type_counts, ensure_ascii=False)}` (globalnie, nie tylko dla odrzuconych rynków). Dla T−59 porównano referencję BBO w {bbo_time_relations['checked_market_entries']:,} kwalifikowanych do tej kontroli wpisach: {bbo_time_relations['older_reference_events']:,} starszych i {bbo_time_relations['tied_reference_events']:,} równych czasowo aktualizacji strony. Rozkład przyczyn odrzuceń i przykłady w `book_rejection_markets.csv`/`BOOK_TRACE_EXAMPLES.md` wskazują na stan inicjalizacji, crossed book, świeżość ask i jakość uzgodnienia BBO; nie ma podstaw, by przypisać je do wariantu schematu.",
        "",
        "Próbki początku, środka i końca okresu: ceny w kolumnie to best bid/best ask, a kwota po średniku to zrekonstruowany VWAP zakupu $5. Czasy ask pokazują odbiór kolektora i czas źródłowy ostatniej zmiany głębokości.",
        "",
        "| Okres | Rynek / condition_id | T | Wejście | Pierwsza obserwacja archiwalna | Inicjalizacja booka UP / DOWN | Ostatnie zdarzenie odebrane przed wejściem | Ostatnia zmiana ask: UP / DOWN (odbiór; źródło; wiek) | Natywne BBO, rozmiar ask i cena $5 |",
        "|---|---|---|---|---|---|---|---|---|",
        *example_lines,
        "",
        "Te archiwalne `timestamp_received` pochodzą od eksportera, nie od naszego przyszłego serwera. Live używa obecnie REST `/book` dla obu tokenów; znacznik odbioru jest lokalny po pobraniu i parsowaniu odpowiedzi, a źródłowy timestamp zostaje pusty, jeśli API go nie zwraca. Kod nie utrzymuje jeszcze strumienia booka. Nie utożsamiam tych zegarów.",
        "",
        "## Porównanie ekonomiczne T−59 s",
        "",
        "Każdy wiersz stosuje te same dostępne snapshoty, filtr ask age ≤30 s, zakup brutto $5, początkową gotówkę $100, model historycznej opłaty i zwrot kapitału 60 s po rozstrzygnięciu. Strategie wybierają transakcje niezależnie; wspólny zbiór oznacza te same kwalifikujące się rynki, a nie wymuszone identyczne transakcje.",
        "",
        "`MARKET_ONLY` nie ma zapisanego, porównywalnego portfela pre-open. Z cache policzono dwa nietrenowane baseline’y bez informacji BTC: stałe p=0.5 i p=0.5015364895 (prewalencja development). Nie stroiłem ich do okresu. Candidate_platt daje na wspólnych rynkach PnL "
        f"{_money(candidate_platt.net_pnl_usd)} wobec {_money(original_platt.net_pnl_usd)} dla oryginalnego Platt; baseline’y dają odpowiednio {_money(constant_baseline.net_pnl_usd)} i {_money(prevalence.net_pnl_usd)}. To dodatni wynik tego replayu, nie potwierdzenie niezależnej przewagi.",
        "",
        "| Zakres | Model | PnL netto | Kapitał końcowy | Drawdown | Transakcje | Obrót | Opłaty | Odrzucenia danych / bez przewagi / brak salda |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *comparison_lines,
        "",
        "## Wpływ walidacji kwotowań na ekonomikę T−59",
        "",
        "Tabela porównuje trzy etapy filtracji przy tych samych zapisanych cenach, fillach, opłatach i zasadach gotówki. Pierwszy odtwarza dawną walidację bid < ask; drugi dopuszcza poprawne bid = ask przy starym uzgadnianiu BBO; trzeci stosuje korektę świeżości referencji. Każdy wiersz pokazuje rynki kwalifikowane w danym etapie. Pełna tabela sześciu modeli/baseline’ów dla wszystkich rynków, zbiorów kwalifikowanych i ich przecięcia oraz lista zmian per rynek są w CSV.",
        "",
        f"Dopuszczenie zablokowanych kwotowań dodało {quote_transition_counts.get('added_by_accepting_locked_quote', 0):,} kwalifikowane rynki przed korektą referencji BBO. Korekta świeżości dodała {quote_transition_counts.get('added_by_fresh_bbo_reconciliation', 0):,} i odrzuciła {quote_transition_counts.get('removed_by_fresh_bbo_reconciliation', 0):,} po potwierdzeniu świeżej rozbieżności. Na wspólnym zbiorze wynik każdego z sześciu modeli jest identyczny we wszystkich trzech etapach.",
        f"Poprzednie +{_money(candidate_common_quote_result.net_pnl_usd)} pozostaje wynikiem na {candidate_common_quote_result.sample_markets:,} rynkach wspólnych dla trzech walidacji; skorygowany zbiór obejmuje {candidate_corrected_quote_result.sample_markets:,} kwalifikowanych rynków i daje {_money(candidate_corrected_quote_result.net_pnl_usd)} dla candidate_platt. Różnica wynika ze zmienionej kwalifikacji snapshotów, nie ze zmiany modelu ani strategii.",
        "",
        "| Walidacja | Rynki kwalifikowane | Model | PnL netto | Transakcje |",
        "|---|---:|---|---:|---:|",
        *quote_impact_lines,
        "",
        f"Księgowanie odtworzono z poprawionych wpisów T−59: gotówka przed wejściem musi pokryć pełny debet (`$5 + fee w collateral`, bez kredytu), kapitał jest blokowany do `resolved_at_utc + 60 s`, a po ostatnim rynku symulator rozlicza wszystkie pozostałe pozycje. W {len(main_trades):,} transakcjach `candidate_platt` znaleziono {negative_cash_count} ujemnych stanów gotówki i {debit_violation_count} naruszeń pokrycia debetu; minimum po wejściu wyniosło {_money(minimum_cash_after)}. Legacy fee zmniejsza liczbę udziałów; współczesna opłata jest debetowana w collateral. Dokładne zaokrąglenie maker-level nie jest dostępne w zagregowanym booku. Drawdown liczy gotówkę plus koszt zablokowanych pozycji, bez mark-to-market.",
        "",
        "Candidate_platt przewyższa oryginalny model Platt i oba proste baseline’y w zapisanej symulacji. Nie istnieje jednak porównywalny wyuczony baseline `MARKET_ONLY` w tej samej pre-open definicji; wcześniejsze wyniki MARKET_ONLY mają inny moment decyzji/feature availability i nie są podstawiane do tej tabeli.",
        "",
        "## Chronologia i niezależność oceny",
        "",
        f"Fit i dobór iteracji kończyły się przed {training['fit_end_exclusive']}; ostatnia dostępna etykieta treningu to {training['latest_label_available_at']}. Kalibracja Platt używała etykiet dostępnych do 2026-04-15 17:00 UTC, a pierwsza decyzja okresu testowego była 17:04 UTC (T rynku 17:05). Kandydat wybierano na wcześniejszych foldach Q3 2025. Nie wykryto bezpośredniego przecieku etykiety do predykcji na badanych punktach czasu.",
        "",
        "Mimo tej chronologii okres ekonomiczny 2026-04-15–2026-05-18 był już użyty w wcześniejszych eksperymentach repozytorium: mieści się w foldach selekcji/tuningu innych komponentów i był raportowany jako development. Wynik ekonomiczny jest więc retrospektywnym wynikiem rozwojowym, a nie niezależnym, nietkniętym holdoutem. Przyczynowość rekonstrukcji booka i niezależność wyboru modelu to odrębne własności.",
        "",
        "Tę tabelę policzono bez dostrajania strategii do okresu. Przyjęto istniejącą regułę dodatniego oczekiwanego zwrotu netto, age cap 30 s, gross $5 i zwolnienie kapitału po 60 s. Nie ma w repozytorium zamrożonego, datowanego przed okresem protokołu potwierdzającego niezależny wybór tych ekonomicznych ustawień; z uwagi na historyczne użycie danych nie traktuję ich wyniku jako prospektywnego testu strategii.",
        "",
        "## Telemetria opóźnień live",
        "",
        "Każda decyzja zachowuje istniejący rekord CSV oraz identyfikator decision/condition/token/model; dołączone są czasy UTC danych Binance (source i receive), gotowości cech, predykcji, decyzji i końca cyklu. Księga zapisuje początek requestu, źródłowy timestamp jeśli istnieje, lokalny odbiór, pochodzenie, stan synchronizacji REST oraz BBO UP/DOWN. Submit zapisuje start/koniec wywołania, osobny lokalny odbiór odpowiedzi, order ID i status/powód pominięcia. Czas lokalnych etapów nadal mierzy monotoniczny `perf_counter`; timestampy zdarzeń są UTC.",
        "",
        "Po cyklu log `[latency_summary]` podaje N, p50/p95/p99, maksimum, liczbę przekroczeń budżetu 1 s i ujemnych różnic dla każdego obserwowalnego etapu względem nominalnego T−60; liczniki rozdzielają cykle, próby, odpowiedzi klienta, order IDs, pola fill zgłoszone w odpowiedzi oraz niezależne rekordy zdarzeń fill. Nie wolno odczytywać mediany wszystkich cykli jako opóźnienia prób zlecenia ani sumować percentyli etapów.",
        "",
        "The authenticated Polymarket user stream records order placement updates and partial/final fills, with REST resync and order/attempt linking. It starts only for enabled live submit. Current official protocol docs and mocked reconnect tests were checked, but py-clob-client-v2 is not installed here, so there was no real handshake or observed exchange ACK/fill. A live order is needed to measure acceptance latency; only actual execution events can establish fill time, price, quantity, and any reported fee. HTTP/client response is not an exchange ACK or fill timestamp.",
        "",
        f"W dotychczasowych, innych runtime’ach: {old_cycle['n']} cykli miało close-to-cycle p50/p95/p99 {old_cycle['p50_ms']:.0f}/{old_cycle['p95_ms']:.0f}/{old_cycle['p99_ms']:.0f} ms; {old_submit['n']} synchronicznych submitów miało p50/p95/p99 {old_submit['p50_ms']:.0f}/{old_submit['p95_ms']:.0f}/{old_submit['p99_ms']:.0f} ms. Dwie kolekcje pre-open miały wszystkie wejścia gotowe {input_offsets[0]:.0f} i {input_offsets[1]:.0f} ms po decyzji i miały wyłączone zlecenia. To odrębne historyczne pomiary, nie podstawa do wyboru T−59 i nie pomiary kandydata end-to-end.",
        "",
        "Instrukcja lokalizacji kolumn, odczytu `[latency_summary]` i rozróżnienia czasu send/ACK/fill jest w [`docs/live_telemetry.md`](../../docs/live_telemetry.md).",
        "",
        "## Gotowość kandydata do live",
        "",
        f"Kandydat ma osobny, nieaktywny paper runtime ze ścieżkami modelu, kalibratora, uporządkowanych 112 cech i konfiguracji historii. Nie zmieniono nazw, pozycji ani definicji cech: lista dokładnie zgadza się z oryginalnym v1, a zmieniły się wyuczony booster/kalibrator i parametry. Audyt mapuje rodziny {json.dumps(feature_families, ensure_ascii=False)} i sprawdza {feature_parity['decision_rows']:,} historycznych decyzji względem rebuildów ograniczonych do chwili decyzji. Wektory miały {feature_parity['feature_mismatches']} różnic cech, {feature_parity['mask_mismatches']} różnic maski i {feature_parity['resume_mismatches']} błędów wznowienia. Na lokalnym CPU p50/p95/p99 wyniosły: warm update {feature_timing['warm_state_update']['p50_ms']:.2f}/{feature_timing['warm_state_update']['p95_ms']:.2f}/{feature_timing['warm_state_update']['p99_ms']:.2f} ms, pełny wektor {feature_timing['warm_full_feature_vector']['p50_ms']:.2f}/{feature_timing['warm_full_feature_vector']['p95_ms']:.2f}/{feature_timing['warm_full_feature_vector']['p99_ms']:.2f} ms, predykcja z Platt {feature_timing['warm_model_predict_plus_platt']['p50_ms']:.2f}/{feature_timing['warm_model_predict_plus_platt']['p95_ms']:.2f}/{feature_timing['warm_model_predict_plus_platt']['p99_ms']:.2f} ms, cała ścieżka update→wektor→predykcja {feature_timing['warm_update_vector_predict_end_to_end']['p50_ms']:.2f}/{feature_timing['warm_update_vector_predict_end_to_end']['p95_ms']:.2f}/{feature_timing['warm_update_vector_predict_end_to_end']['p99_ms']:.2f} ms. Szczyt RSS próbkowany co 100 ms: {feature_memory['rss_peak_sampled_bytes'] / (1024**2):.0f} MiB. To potwierdza zgodność badanego lokalnego runtime, nie sprawdza bieżącego live feedu ani złożenia zlecenia. Kandydata nie aktywowano.",
        "",
        f"The exact 44/112 was a name intersection: the active BTC model bundle has {feature_definition_audit['active_general_runtime_feature_count']} columns, of which {feature_definition_audit['candidate_names_present_in_active_general_runtime']} names occur in the candidate; the other {len(feature_definition_audit['candidate_names_absent_from_active_general_runtime'])} are absent from that separate bundle, not unsupported code. The 29-feature pre-open baseline is a separate causal raw-candle model ({feature_definition_audit['29_feature_preopen_baseline_feature_count']} features; {feature_definition_audit['candidate_names_present_in_29_feature_preopen_baseline']} exact name overlap). Candidate and original have the same ordered feature list ({feature_definition_audit['same_ordered_feature_list']}), indicator-fit directory ({feature_definition_audit['indicator_fit_results_dir_same']}), and volume/reaction profile configs ({feature_definition_audit['volume_profile_config_same']}/{feature_definition_audit['reaction_profile_config_same']}); no names, positions, or definitions changed.",
        f"Chaikin SHMMA accumulated numerical drift over more than 3 million candles. The fix serializes and incrementally advances its recurrence state; it does not reset a short window or relax tolerance. The candidate seed is valid through {state_seed_as_of}; a later startup requires a complete contiguous closed-candle catch-up and fails before prediction if that interval is missing.",
        "## Artefakty",
        "",
        "- `primary_economic_comparison.csv` — T−59, wszystkie rynki i wspólny zbiór kwalifikujących się rynków.",
        "- `book_timing_examples.csv` — audyt trzech ksiąg T−59 z archiwum lokalnego.",
        "- `economic_scenarios.csv` — wcześniejsza macierz wariantów czasowych; nie przeliczono jej po korekcie BBO.",
        "- `quote_validation_economic_impact.csv` i `quote_validation_market_changes.csv` — wpływ trzech etapów walidacji BBO na ekonomikę i kwalifikację per rynek.",
        "- `quote_validation_corrected_trades.parquet` — transakcje z finalnej T−59 walidacji dla wszystkich sześciu modeli/baseline’ów.",
        "- `fresh_bbo_replay_summary.json`, `quote_validation_before_after.csv` i `fresh_bbo_entry_results.csv` — zakres, zasoby i wyniki ukierunkowanego replayu świeżości BBO.",
        "- `runtime_compatibility_audit.json`, `feature_compatibility_112.csv`, `feature_definition_comparison.json`, `artifact_manifest.json` and `archive_partition_hashes.csv` - candidate feature map, numerical parity and reproducibility fingerprints.",
        "- `book_rejection_markets.csv`, `book_rejection_daily.csv` i `BOOK_TRACE_EXAMPLES.md` — powody odrzuceń i przykładowe ścieżki księgi.",
        "- `audit.json`, `data_coverage.csv`, `quote_validation_corrected_trades.parquet`, `model_comparison.csv` i `report_bundle.zip` — szczegóły oraz odtwarzalność finalnej walidacji.",
        "",
        "Nie aktywowano kandydata ani nie złożono rzeczywistych zleceń; dodane ścieżki konfiguracji dotyczą osobnego profilu paper.",
        "",
    ]


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
    fresh_bbo = _json(REPORT_DIR / "fresh_bbo_replay_summary.json")
    compatibility = _json(REPORT_DIR / "runtime_compatibility_audit.json")
    extraction = _json(EXTRACT_DIR / "extraction_summary.json")
    checkpoint = _json(EXTRACT_DIR / "hour_checkpoint.json")
    coverage = pd.read_csv(COVERAGE_CSV)
    economic = pd.read_csv(ECONOMIC_SCENARIOS)
    model_comparison = pd.read_csv(MODEL_COMPARISON)
    trades = pd.read_parquet(TRADES)
    market_index = pd.read_parquet(EXTRACT_DIR / "market_index.parquet")
    primary_comparison = _build_primary_economic_comparison(coverage)
    quote_impact, quote_changes, corrected_trades = _build_quote_validation_economic_impact(coverage)
    book_examples = pd.read_csv(BOOK_TIMING_EXAMPLES)

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
            "interpretation": "Retained historical timing context from a different runtime. The current T-59 primary is an accepted operational assumption and is not derived from a measured maximum or guaranteed worst case.",
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
            "candidate_feature_count": compatibility["feature_map_rows"],
            "candidate_runtime_manifest": compatibility["candidate_runtime_manifest"],
            "candidate_runtime_profile_is_active": False,
            "feature_order_exactly_matches_original_v1": compatibility["feature_order_exactly_matches_original"],
            "historical_feature_parity_status": compatibility["feature_parity"]["status"],
            "historical_parity_decision_rows": compatibility["feature_parity"]["decision_rows"],
            "historical_parity_feature_mismatches": compatibility["feature_parity"]["feature_mismatches"],
            "historical_parity_mask_mismatches": compatibility["feature_parity"]["mask_mismatches"],
            "historical_parity_resumption_mismatches": compatibility["feature_parity"]["resume_mismatches"],
            "historical_timing_and_memory_artifact": "reports/btc_preopen/live_feature_parity.json",
            "feature_definition_comparison": compatibility["candidate_feature_definition_audit"],
            "prior_44_of_112_claim_explanation": compatibility[
                "candidate_feature_definition_audit"
            ]["prior_44_of_112_claim_explanation"],
            "feature_values_compared": compatibility["feature_parity"]["comparison"]["feature_values_compared"],
            "max_feature_abs_delta": compatibility["feature_parity"]["comparison"]["feature_max_abs_delta"],
            "feature_tolerances": compatibility["feature_parity"]["predeclared_tolerances"],
            "local_py_clob_client_v2_installed": False,
            "generic_256_column_runtime_overlap_is_not_a_candidate_readiness_measure": True,
            "reason": "The inactive paper profile is wired to the candidate booster, calibrator, 112-column order, history configuration, and validated Chaikin recurrence seed. Historical causal feature parity was checked against the saved candidate dataset on local CPU; startup must fetch a complete contiguous candle catch-up from the seed timestamp. No current exchange feed or live order was exercised. User-stream protocol and REST resync were checked against current official documentation and mocks, but the local py-clob-client-v2 package is absent, so no real client handshake was possible.",
        },
        "interpretation": [
            "The June historical log supports sub-second median receive-to-decision behavior in its different live model; the recorded p95/p99 are above one second.",
            "In the historical runtime, live_minute_opened is candle Opened plus one minute, so the WS 'minute open' and next target-window start coincide with the just-closed candle boundary. The wall-clock delay columns are close-anchored for that cycle, but have no recorded clock-offset calibration. Local feature/inference and submit-call durations are separate monotonic measurements and must not be summed as if their quantiles aligned.",
            "Synchronous submit duration includes the client response and is not a separately timestamped exchange acknowledgement or fill. The authenticated Polymarket user-stream logger is now wired for future enabled live runs, but it produced no observed events in this audit.",
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
            "event_ordering_limit": "PMXT rows expose source/receive timestamps and transaction_hash but no monotonic event sequence ID or observed schema-version field. Reconstruction sorts by receive time, source time, market, token, and event type; exact ties retain the order in the filtered Parquet part, but PMXT does not document that row order as an event sequence. A BBO reference is decisive only when its source timestamp is strictly newer than the latest update to both side states; stale or tied references are counted separately, while a fresh ask-side disagreement at entry disqualifies the snapshot.",
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
            "legacy_trade_log_rows_before_fresh_bbo_correction": int(len(trades)),
            "corrected_trade_log_rows": int(len(corrected_trades)),
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
    impact_summary = {
        stage: {
            "eligible_markets": int(group.eligible_markets_at_stage.iloc[0]),
            "stage_eligible_model_rows": group.loc[
                group.sample_scope.eq("stage_eligible_markets")
            ].to_dict("records"),
            "common_eligible_model_rows": group.loc[
                group.sample_scope.eq("common_eligible_across_stages")
            ].to_dict("records"),
        }
        for stage, group in quote_impact.groupby("validation_stage", sort=False)
    }
    audit["candidate_study"]["runtime_compatibility_audit"] = compatibility
    audit["archive_and_replay"]["fresh_bbo_validation"] = fresh_bbo
    audit["archive_and_replay"]["quote_validation_economic_impact"] = {
        "artifacts": [QUOTE_IMPACT.relative_to(ROOT).as_posix(), QUOTE_MARKET_CHANGES.relative_to(ROOT).as_posix()],
        "stages": impact_summary,
        "changed_market_rows_by_transition": {
            str(key): int(value)
            for key, value in quote_changes.validation_transition.value_counts().items()
        },
        "earlier_economic_scenario_matrix_recomputed_after_freshness_fix": False,
    }
    economics["quote_validation_economic_impact"] = {
        "artifacts": [QUOTE_IMPACT.relative_to(ROOT).as_posix(), QUOTE_MARKET_CHANGES.relative_to(ROOT).as_posix()],
        "stages": impact_summary,
        "changed_market_rows_by_transition": {
            str(key): int(value)
            for key, value in quote_changes.validation_transition.value_counts().items()
        },
        "earlier_scenario_matrix_status": "retained from before the targeted fresh-BBO correction; use quote_validation_economic_impact.csv for same-T-59 stage comparisons",
    }
    economics["corrected_trade_log"] = {
        "artifact": CORRECTED_TRADES.relative_to(ROOT).as_posix(),
        "rows": int(len(corrected_trades)),
        "models": sorted(corrected_trades.model.unique().tolist()),
        "validation_stage": "locked_quotes_corrected_bbo_freshness",
    }
    (REPORT_DIR / "audit.json").write_text(json.dumps(_json_safe(audit), indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")

    main_case = "prestart_c0_o1"
    primary_cases = [main_case, "prestart_c0_o0", "prestart_c0_o2", "prestart_c0_o5"]
    primary_scenarios = {
        case: _get_scenario(economic, case, 30, 60) for case in primary_cases
    }
    robustness_cases = ["prestart_c15_o5", "prestart_c45_o5"]
    robustness = {
        case: _get_scenario(economic, case, 30, 60) for case in robustness_cases
    }
    economics["primary_scenario"] = "prestart_c0_o1|age30|release60 (fixed accepted operational assumption: entry T-59s, one second after nominal T-60s; not a measured maximum or worst case)"
    economics["main_entry_times_relative_to_market_start"] = {
        "prestart_c0_o1": "T-59s (fixed accepted operational assumption)",
        "prestart_c0_o0": "T-60s (retained prior ideal reference)",
        "prestart_c0_o2": "T-58s (retained prior sensitivity)",
        "prestart_c0_o5": "T-55s (retained prior local component scenario)",
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
    audit["archive_and_replay"]["primary_economic_comparison"] = {
        "artifact": ECONOMIC_COMPARISON.relative_to(ROOT).as_posix(),
        "rows": primary_comparison.to_dict("records"),
        "method": "Cached T-59 entry prices and fills with corrected fresh-BBO eligibility; same $5 gross order, fee model, $100 initial cash, and 60-second post-resolution capital release. Constant-probability baselines are not tuned.",
        "corrected_trade_log": CORRECTED_TRADES.relative_to(ROOT).as_posix(),
    }
    audit["archive_and_replay"]["primary_entry_book_examples"] = [
        {
            **row._asdict(),
            "up_token_id": str(market_index.loc[
                market_index.condition_id.eq(row.condition_id), "up_token_id"
            ].iloc[0]),
            "down_token_id": str(market_index.loc[
                market_index.condition_id.eq(row.condition_id), "down_token_id"
            ].iloc[0]),
        }
        for row in book_examples.itertuples(index=False)
    ]
    (REPORT_DIR / "audit.json").write_text(
        json.dumps(_json_safe(audit), indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    report_lines = _build_completed_summary(
        examples=book_examples,
        market_index=market_index,
        coverage=coverage,
        comparison=primary_comparison,
        quote_impact=quote_impact,
        quote_changes=quote_changes,
        compatibility=compatibility,
        trades=corrected_trades,
        live_timing=live_timing,
        training=training,
    )
    report_lines = [_repair_report_line(line) for line in report_lines]
    (REPORT_DIR / "SUMMARY.md").write_text("\n".join(report_lines), encoding="utf-8")

    include_paths = [
        REPORT_DIR / "SUMMARY.md", REPORT_DIR / "audit.json", COVERAGE_CSV,
        MODEL_COMPARISON, ECONOMIC_SCENARIOS, ECONOMIC_REPLAY, ECONOMIC_COMPARISON,
        QUOTE_IMPACT, QUOTE_MARKET_CHANGES, CORRECTED_TRADES,
        BOOK_TIMING_EXAMPLES,
        CANDIDATE_METRICS, REPORT_DIR / "candidate_search_trials.csv",
        INFERENCE_LATENCY, ROOT / "benchmark_btc_preopen_bundle.py",
        REPORT_DIR / "candidate_external_predictions.parquet",
        REPORT_DIR / "original_bundle_verification.json",
        REPORT_DIR / "runtime_compatibility_audit.json",
        REPORT_DIR / "live_feature_parity.json",
        REPORT_DIR / "live_feature_parity_by_feature.csv",
        REPORT_DIR / "live_feature_parity_anchors.csv",
        REPORT_DIR / "feature_compatibility_112.csv",
        REPORT_DIR / "feature_definition_comparison.json",
        ROOT / "configs/runtime/btc_preopen_candidate.json",
        ROOT / "configs/runtime/btc_preopen_candidate_features.json",
        ROOT / "configs/runtime/btc_preopen_candidate_history_requirements.json",
        ROOT / "configs/runtime/btc_preopen_candidate_model_meta.json",
        ROOT / "configs/runtime/btc_preopen_candidate_indicator_state.json",
        ROOT / "data/models/BTC/20261003_043549/lgbm_meta_20261003_043549.json",
        ROOT / "data/models/BTC/btc_preopen_v1/lgbm_meta.json",
        REPORT_DIR / "book_rejection_markets.csv",
        REPORT_DIR / "book_rejection_daily.csv",
        REPORT_DIR / "BOOK_TRACE_EXAMPLES.md",
        REPORT_DIR / "fresh_bbo_entry_results.csv",
        REPORT_DIR / "fresh_bbo_replay_summary.json",
        REPORT_DIR / "quote_validation_before_fresh_bbo.csv",
        REPORT_DIR / "quote_validation_before_after.csv",
        REPORT_DIR / "book_validation_before.json",
        REPORT_DIR / "artifact_manifest.json",
        REPORT_DIR / "archive_partition_hashes.csv",
        candidate_calibrator_path,
        ROOT / "data/analysis/polymarket/BTC/preopen_v1/candidate_study_20261005/study_identity.json",
        ROOT / "data/analysis/polymarket/BTC/preopen_v1/candidate_study_20261005/frozen_candidate_list.json",
        ROOT / "data/analysis/polymarket/BTC/preopen_v1/candidate_study_20261005/selection_results.json",
        *sorted(REPORT_DIR.glob("point_in_time_*.json")),
        ORIGINAL_TRAINING_MANIFEST, ORIGINAL_EVALUATION, ORIGINAL_MODEL_TUNING,
        ORIGINAL_CONFIG, EXTRACT_DIR / "extraction_summary.json", EXTRACT_DIR / "archive_identity.json",
        ROOT / "docs/polymarket_btc_experiment.md", ROOT / "docs/live_telemetry.md",
        ROOT / "docs/btc_preopen_experiment_manifest_20261004.json",
        ROOT / "run.py", ROOT / "utils/live.py", ROOT / "utils/polymarket_user_stream.py", ROOT / "README.md",
        ROOT / "audit_btc_oof.py",
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
        ROOT / "tests/test_btc_preopen_pmxt_extract.py", ROOT / "tests/test_live_utils.py",
        ROOT / "tests/test_run_multi_asset_latency.py",
        ROOT / "tests/test_polymarket_user_stream.py",
        ROOT / "audit_btc_preopen_runtime_compatibility.py",
        ROOT / "audit_feature_readiness.py", ROOT / "utils/project_config.py",
    ]
    include_paths = list(dict.fromkeys(include_paths))
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
