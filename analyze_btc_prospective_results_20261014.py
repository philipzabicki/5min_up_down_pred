"""Score the frozen prospective BTC-plus-market test after its fixed deadline."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import run_btc_prospective_shadow as prospective
import run_btc_preopen_t59_reassessment as economic_replay


ROOT = Path(__file__).resolve().parent
DB_PATH = prospective.DB_PATH
OUTPUT_DIR = ROOT / "reports/btc_prospective_test_20261014"
REPLICATES = 5000
SEED = 20261014
BLOCK_DAYS = 3
INITIAL_PORTFOLIO_USD = float(prospective.PROTOCOL["economics"]["portfolio_initial_free_cash_usd"])
PERCENT_OF_FREE_CASH = 0.05
def _read_session(connection):
    row = connection.execute(
        "SELECT started_at_utc,first_market_start_utc,evaluation_deadline_utc,protocol_sha256,artifact_manifest_json FROM session WHERE id=1"
    ).fetchone()
    if row is None:
        raise RuntimeError("Prospective database has no frozen session record")
    return {
        "started_at_utc": row[0], "first_market_start_utc": row[1],
        "evaluation_deadline_utc": row[2], "protocol_sha256": row[3],
        "artifact_sha256": json.loads(row[4]),
    }


def _read_attempts(connection):
    rows = connection.execute(
        "SELECT market_slug,market_start_utc,attempt_status,reason,t60_snapshot_json,t59_snapshot_json,probabilities_json,policy_json,timings_json FROM market_attempts ORDER BY market_start_utc"
    ).fetchall()
    columns = (
        "market_slug", "market_start_utc", "attempt_status", "reason", "t60_snapshot",
        "t59_snapshot", "probabilities", "policy", "timings",
    )
    result = []
    for row in rows:
        record = dict(zip(columns, row))
        for key in columns[4:]:
            record[key] = json.loads(record[key]) if record[key] else None
        record["market_start_utc"] = pd.to_datetime(record["market_start_utc"], utc=True)
        result.append(record)
    return result


def _read_outcomes(connection):
    rows = connection.execute(
        "SELECT market_slug,condition_id,target_up,resolved_at_utc,retrieved_at_utc,payload_json FROM official_outcomes"
    ).fetchall()
    return {
        row[0]: {
            "market_slug": row[0], "condition_id": row[1], "target_up": int(row[2]),
            "resolved_at_utc": row[3], "retrieved_at_utc": row[4],
            "payload": json.loads(row[5]),
        }
        for row in rows
    }


def _latency_summary(values):
    values = np.asarray([float(value) for value in values if value is not None and np.isfinite(float(value))])
    if not len(values):
        return {"count": 0, "median_seconds": None, "p95_seconds": None, "max_seconds": None}
    return {
        "count": int(len(values)),
        "median_seconds": float(np.median(values)),
        "p95_seconds": float(np.quantile(values, 0.95)),
        "max_seconds": float(np.max(values)),
    }


def _collection_status(now, market_deadline, outcome_deadline):
    now = pd.to_datetime(now, utc=True)
    market_deadline = pd.to_datetime(market_deadline, utc=True)
    outcome_deadline = pd.to_datetime(outcome_deadline, utc=True)
    if now < market_deadline:
        return "collection_in_progress"
    if now < outcome_deadline:
        return "settlement_collection_in_progress"
    return "complete"


def _scheduled_markets(first_market, deadline):
    return pd.date_range(
        pd.to_datetime(first_market, utc=True),
        pd.to_datetime(deadline, utc=True),
        freq="5min",
        inclusive="left",
    )


def _log_loss(outcomes, probabilities):
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(outcomes, dtype=np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def _paired_block_interval(frame, difference_col, all_days):
    daily = frame.assign(
        day=frame["market_start_utc"].dt.floor("D")
    ).groupby("day")[difference_col].agg(["sum", "count"])
    daily = daily.reindex(all_days, fill_value=0)
    sums = daily["sum"].to_numpy(dtype=np.float64)
    counts = daily["count"].to_numpy(dtype=np.float64)
    block_count = int(np.ceil(len(all_days) / BLOCK_DAYS))
    rng = np.random.default_rng(SEED)
    draws = np.empty(REPLICATES, dtype=np.float64)
    for rep in range(REPLICATES):
        starts = rng.integers(0, len(all_days), size=block_count)
        positions = np.concatenate([
            (np.arange(start, start + BLOCK_DAYS) % len(all_days)) for start in starts
        ])[:len(all_days)]
        n = float(counts[positions].sum())
        draws[rep] = float(sums[positions].sum() / n) if n else np.nan
    draws = draws[np.isfinite(draws)]
    return {
        "mean_paired_difference": float(frame[difference_col].mean()),
        "ci_95_two_sided": np.quantile(draws, [0.025, 0.975]).tolist(),
        "block_days": BLOCK_DAYS,
        "replicates": int(len(draws)),
        "seed": SEED,
        "sampled_utc_days": int(len(all_days)),
    }


def _book_metadata(snapshot):
    return {
        "tokens": {
            "up": snapshot["up_book"]["token_id"],
            "down": snapshot["down_book"]["token_id"],
        },
        "slug": snapshot["market_slug"],
        "condition_id": snapshot["condition_id"],
    }


def _market_for_policy(snapshot, p_a, stake):
    metadata = _book_metadata(snapshot)
    return {
        "market_slug": metadata["slug"],
        "condition_id": metadata["condition_id"],
        "market_start_utc": snapshot["gamma_market_payload"].get("eventStartTime")
        or snapshot["market_start_utc"],
        "up_token_id": metadata["tokens"]["up"],
        "down_token_id": metadata["tokens"]["down"],
        "order_min_size_usd_current": float(snapshot["order_min_size_usd"] or 0.0),
        "p_candidate_platt": float(p_a),
        "gross_order_usd": float(stake),
    }


def _dynamic_portfolio(rows, outcomes, outcome_deadline):
    free_cash = INITIAL_PORTFOLIO_USD
    locked = 0.0
    pending = []
    fills = []
    skipped = Counter()
    peak_equity = INITIAL_PORTFOLIO_USD
    max_drawdown = 0.0
    for attempt in rows:
        now = attempt["market_start_utc"] - pd.Timedelta(seconds=60)
        due, pending = [x for x in pending if x["retrieved_at"] <= now], [x for x in pending if x["retrieved_at"] > now]
        for item in due:
            if item["payout"] is not None:
                free_cash += item["payout"]
            locked -= item["debit"]
        equity = free_cash + locked
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            max_drawdown = max(max_drawdown, 1.0 - equity / peak_equity)
        if attempt["attempt_status"] != "complete":
            skipped["incomplete_t60_decision"] += 1
            continue
        outcome = outcomes.get(attempt["market_slug"])
        t60, t59 = attempt.get("t60_snapshot"), attempt.get("t59_snapshot")
        probabilities = attempt.get("probabilities") or {}
        if not t60 or not t59 or not t60.get("quote_valid") or not t59.get("quote_valid"):
            skipped["missing_execution_book"] += 1
            continue
        stake = free_cash * PERCENT_OF_FREE_CASH
        if stake <= 0.0:
            skipped["no_free_cash"] += 1
            continue
        metadata = _book_metadata(t60)
        market = _market_for_policy(t60, probabilities["p_A_frozen"], stake)
        decision = economic_replay._evaluate_market_sides(
            market, prospective._execution_snapshot(t60, metadata), stake
        )
        side = decision["chosen_side"]
        if side is None or decision["chosen_ev_usd"] is None or decision["chosen_ev_usd"] < 0.01 * stake:
            skipped["below_t60_net_ev_hurdle"] += 1
            continue
        market["market_start_utc"] = attempt["market_start_utc"].isoformat()
        execution = economic_replay._price_side(
            prospective._execution_snapshot(t59, metadata), market, side, stake
        )
        fill = execution.get("fill") if execution.get("priceable") else None
        if fill is None:
            skipped["t59_unpriceable"] += 1
            continue
        debit = float(fill["cash_debit_usd"])
        if debit > free_cash + 1e-9:
            skipped["would_require_credit"] += 1
            continue
        free_cash -= debit
        locked += debit
        outcome_available_by_deadline = bool(
            outcome and pd.to_datetime(outcome["retrieved_at_utc"], utc=True)
            <= pd.to_datetime(outcome_deadline, utc=True)
        )
        if outcome_available_by_deadline:
            payout = float(fill["shares"]) * (int(outcome["target_up"]) if side == "up" else 1 - int(outcome["target_up"]))
            settle_at = pd.to_datetime(outcome["retrieved_at_utc"], utc=True)
        else:
            payout = None
            settle_at = pd.to_datetime(outcome_deadline, utc=True) + pd.Timedelta(nanoseconds=1)
            skipped["position_unresolved_at_settlement_deadline"] += 1
        pending.append({"retrieved_at": settle_at, "debit": debit, "payout": payout})
        fills.append({
            "market_slug": attempt["market_slug"], "side": side, "stake_usd": stake,
            "cash_debit_usd": debit, "payout_usd": payout,
            "realized_net_pnl_usd": None if payout is None else payout - debit,
            "settled_by_deadline": outcome_available_by_deadline,
            "execution_assumption": execution["execution_assumption"],
        })
        equity = free_cash + locked
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            max_drawdown = max(max_drawdown, 1.0 - equity / peak_equity)
    final_time = pd.to_datetime(outcome_deadline, utc=True)
    due, pending = [x for x in pending if x["retrieved_at"] <= final_time], [x for x in pending if x["retrieved_at"] > final_time]
    for item in due:
        if item["payout"] is not None:
            free_cash += item["payout"]
        locked -= item["debit"]
    return {
        "initial_free_cash_usd": INITIAL_PORTFOLIO_USD,
        "sizing": "5% of free cash; no credit; original order minimum retained",
        "settlement_cash_release_time": "first Gamma outcome retrieval time, conservatively later than resolved_at when applicable",
        "fills": len(fills),
        "unresolved_positions_at_settlement_deadline": len(pending),
        "realized_net_pnl_usd": float(sum(x["realized_net_pnl_usd"] or 0.0 for x in fills if x["settled_by_deadline"])),
        "free_cash_at_settlement_deadline_usd": float(free_cash),
        "locked_cost_basis_at_settlement_deadline_usd": float(max(locked, 0.0)),
        "max_drawdown_on_cost_basis": float(max_drawdown),
        "skip_reasons": dict(skipped),
        "actual_fill": "unknown",
        "actual_fee": "unknown",
    }


def _economic_summary(rows, outcomes, outcome_deadline):
    outcome_deadline = pd.to_datetime(outcome_deadline, utc=True)
    fixed = {"eligible_policy_signals": 0, "priceable_t59_fills": 0, "missing_outcome": 0, "realized_net_pnl_usd": 0.0, "gross_turnover_usd": 0.0, "fee_usd": 0.0, "active_utc_days": set(), "execution_assumptions": Counter()}
    for attempt in rows:
        if attempt["attempt_status"] != "complete":
            continue
        policy = attempt.get("policy") or {}
        if policy.get("selected_side") is None:
            continue
        fixed["eligible_policy_signals"] += 1
        execution = ((policy.get("t59_execution_observation") or {}).get("simulated_execution") or {})
        fill = execution.get("fill") if execution.get("priceable") else None
        if fill is None:
            continue
        fixed["priceable_t59_fills"] += 1
        fixed["execution_assumptions"][execution.get("execution_assumption", "unknown")] += 1
        outcome = outcomes.get(attempt["market_slug"])
        if not outcome or pd.to_datetime(outcome["retrieved_at_utc"], utc=True) > outcome_deadline:
            fixed["missing_outcome"] += 1
            continue
        target = int(outcome["target_up"])
        selected_outcome = target if policy["selected_side"] == "up" else 1 - target
        pnl = float(fill["shares"]) * selected_outcome - float(fill["cash_debit_usd"])
        fixed["realized_net_pnl_usd"] += pnl
        fixed["gross_turnover_usd"] += float(fill["gross_usd"])
        fixed["fee_usd"] += float(fill["fee_usd"])
        fixed["active_utc_days"].add(attempt["market_start_utc"].floor("D").date().isoformat())
    fixed["active_utc_days"] = len(fixed["active_utc_days"])
    fixed["execution_assumptions"] = dict(fixed["execution_assumptions"])
    fixed["fixed_policy"] = "A_frozen, T-60 1% net-EV hurdle, independent $5 gross orders"
    fixed["actual_fill"] = "unknown"
    fixed["actual_fee"] = "unknown"
    return {
        "fixed_5_usd": fixed,
        "five_percent_free_cash": _dynamic_portfolio(rows, outcomes, outcome_deadline),
        "no_trade_control_net_pnl_usd": 0.0,
        "quality_sample_filters_used": False,
    }


def _operational_summary(connection, attempts, outcomes, scheduled, outcome_deadline):
    scheduled_slugs = {f"{prospective.MARKET_PREFIX}-{int(stamp.timestamp())}" for stamp in scheduled}
    attempts = [x for x in attempts if x["market_slug"] in scheduled_slugs]
    attempt_map = {x["market_slug"]: x for x in attempts}
    status = Counter(x["attempt_status"] for x in attempts)
    latencies = {
        "t60_deadline_to_quality_inputs_ready_seconds": [],
        "t60_deadline_to_btc_candle_receipt_seconds": [],
        "feature_construction_seconds": [],
        "prediction_and_calibration_seconds": [],
        "decision_computation_seconds": [],
        "t60_book_request_seconds": [],
        "t59_book_request_seconds": [],
    }
    btc_sources = Counter()
    disconnected_feed_attempts = 0
    for attempt in attempts:
        timings = attempt.get("timings") or {}
        if timings.get("t60_quality_inputs_ready_at_utc"):
            ready_at = pd.to_datetime(timings["t60_quality_inputs_ready_at_utc"], utc=True)
            deadline_at = attempt["market_start_utc"] - pd.Timedelta(minutes=1)
            latencies["t60_deadline_to_quality_inputs_ready_seconds"].append((ready_at - deadline_at).total_seconds())
        if timings.get("btc_data_received_at_utc"):
            received_at = pd.to_datetime(timings["btc_data_received_at_utc"], utc=True)
            deadline_at = attempt["market_start_utc"] - pd.Timedelta(minutes=1)
            latencies["t60_deadline_to_btc_candle_receipt_seconds"].append((received_at - deadline_at).total_seconds())
        source = (timings.get("btc_source_metadata") or {}).get("source", "unavailable")
        btc_sources[source] += 1
        feed_states = timings.get("btc_websocket_connections") or {}
        if feed_states and any(not state.get("connected", False) for state in feed_states.values() if "connected" in state):
            disconnected_feed_attempts += 1
        mapped_values = {
            "feature_construction_seconds": timings.get("feature_construction_seconds"),
            "prediction_and_calibration_seconds": timings.get("prediction_and_calibration_seconds"),
            "decision_computation_seconds": timings.get("decision_computation_seconds"),
            "t60_book_request_seconds": (attempt.get("t60_snapshot") or {}).get("request_duration_seconds"),
            "t59_book_request_seconds": timings.get("t59_snapshot_request_seconds"),
        }
        for key, value in mapped_values.items():
            if value is not None:
                latencies[key].append(value)
    outcome_count = sum(
        slug in outcomes and pd.to_datetime(outcomes[slug]["retrieved_at_utc"], utc=True) <= pd.to_datetime(outcome_deadline, utc=True)
        for slug in attempt_map
    )
    return {
        "scheduled_market_count": int(len(scheduled)),
        "append_only_attempt_count": int(len(attempts)),
        "attempt_coverage": float(len(attempts) / len(scheduled)) if len(scheduled) else None,
        "unrecorded_scheduled_markets": int(max(len(scheduled) - len(attempts), 0)),
        "attempt_status_counts": dict(status),
        "btc_candle_source_counts": dict(btc_sources),
        "attempts_with_disconnected_btc_feed": int(disconnected_feed_attempts),
        "late_or_missing_attempts": int(sum(v for k, v in status.items() if k != "complete")),
        "official_outcomes_available_by_settlement_deadline": int(outcome_count),
        "missing_official_outcomes_by_settlement_deadline": int(len(attempts) - outcome_count),
        "outcome_retrieval_latency_seconds": [
            (pd.to_datetime(outcomes[slug]["retrieved_at_utc"], utc=True)
             - pd.to_datetime(outcomes[slug]["resolved_at_utc"], utc=True)).total_seconds()
            for slug in attempt_map if slug in outcomes and outcomes[slug]["resolved_at_utc"]
             and pd.to_datetime(outcomes[slug]["retrieved_at_utc"], utc=True) <= pd.to_datetime(outcome_deadline, utc=True)
        ],
        "latency_seconds": {
            **{key: _latency_summary(values) for key, values in latencies.items()},
            "official_outcome_retrieval_after_resolution_seconds": _latency_summary([
                (pd.to_datetime(outcomes[slug]["retrieved_at_utc"], utc=True)
                 - pd.to_datetime(outcomes[slug]["resolved_at_utc"], utc=True)).total_seconds()
                for slug in attempt_map if slug in outcomes and outcomes[slug]["resolved_at_utc"]
                and pd.to_datetime(outcomes[slug]["retrieved_at_utc"], utc=True) <= pd.to_datetime(outcome_deadline, utc=True)
            ]),
        },
        "t59_book_valid_count": int(sum(bool((x.get("t59_snapshot") or {}).get("quote_valid")) for x in attempts)),
    }


def _score(attempts, outcomes, scheduled, outcome_deadline, protocol):
    scheduled_slugs = {f"{prospective.MARKET_PREFIX}-{int(stamp.timestamp())}" for stamp in scheduled}
    by_slug = {x["market_slug"]: x for x in attempts if x["market_slug"] in scheduled_slugs}
    quality = []
    exclusions = Counter()
    for market_start in scheduled:
        slug = f"{prospective.MARKET_PREFIX}-{int(market_start.timestamp())}"
        attempt = by_slug.get(slug)
        outcome = outcomes.get(slug)
        if attempt is None:
            exclusions["missing_attempt_record"] += 1
            continue
        if not outcome or pd.to_datetime(outcome["retrieved_at_utc"], utc=True) > pd.to_datetime(outcome_deadline, utc=True):
            exclusions["missing_official_outcome_by_settlement_deadline"] += 1
            continue
        if attempt.get("attempt_status") != "complete":
            exclusions[f"attempt_status:{attempt.get('attempt_status')}"] += 1
            continue
        if not (attempt.get("t60_snapshot") or {}).get("quote_valid"):
            exclusions["invalid_t60_market_book"] += 1
            continue
        if not (attempt.get("timings") or {}).get("quality_inputs_ready_before_t59"):
            exclusions["t60_forecast_late_or_missing"] += 1
            continue
        if prospective._quality_row(attempt, outcome):
            quality.append({
                "market_slug": slug,
                "market_start_utc": attempt["market_start_utc"],
                "target_up": int(outcome["target_up"]),
                **attempt["probabilities"],
            })
        else:
            exclusions["invalid_or_missing_probability"] += 1
    frame = pd.DataFrame(quality)
    if frame.empty:
        return {
            "common_quality_market_count": 0,
            "quality_days_with_at_least_100_markets": 0,
            "coverage_gate_passed": False,
            "primary_verdict": "inconclusive",
            "quality_exclusion_counts": dict(exclusions),
            "primary": None,
            "secondary": None,
            "benchmark_log_loss": {},
        }
    frame["market_start_utc"] = pd.to_datetime(frame["market_start_utc"], utc=True)
    for prediction in ("p_market", "p_A_frozen", "p_C_market_only", "p_D_market_plus_btc"):
        frame[f"ll_{prediction}"] = _log_loss(frame.target_up, frame[prediction])
        frame[f"brier_{prediction}"] = (frame.target_up - frame[prediction]) ** 2
    calibration = {}
    bins = np.linspace(0.0, 1.0, 11)
    for prediction in ("p_market", "p_A_frozen", "p_C_market_only", "p_D_market_plus_btc"):
        probability = frame[prediction].to_numpy(dtype=float)
        outcome = frame.target_up.to_numpy(dtype=float)
        bucket = np.minimum(np.digitize(probability, bins[1:-1], right=False), 9)
        ece = 0.0
        for index in range(10):
            selected = bucket == index
            if selected.any():
                ece += float(selected.mean()) * abs(float(probability[selected].mean() - outcome[selected].mean()))
        calibration[prediction] = {
            "mean_probability_minus_event_rate": float(probability.mean() - outcome.mean()),
            "fixed_10_bin_expected_calibration_error": float(ece),
        }
    frame["primary_difference"] = frame["ll_p_D_market_plus_btc"] - frame["ll_p_market"]
    frame["secondary_difference"] = frame["ll_p_D_market_plus_btc"] - frame["ll_p_C_market_only"]
    first_day = pd.to_datetime(scheduled[0], utc=True).floor("D")
    all_days = pd.date_range(first_day, periods=int(protocol["quality_test"]["fixed_calendar_days"]), freq="D")
    daily_count = frame.assign(day=frame.market_start_utc.dt.floor("D")).groupby("day").size().reindex(all_days, fill_value=0)
    quality_days = int(daily_count.ge(100).sum())
    primary_result = _paired_block_interval(frame, "primary_difference", all_days)
    secondary_result = _paired_block_interval(frame, "secondary_difference", all_days)
    coverage_gate = bool(
        len(frame) >= int(protocol["quality_test"]["minimum_common_quality_markets"])
        and quality_days >= int(protocol["quality_test"]["minimum_quality_days"])
    )
    return {
        "common_quality_market_count": int(len(frame)),
        "quality_exclusion_counts": dict(exclusions),
        "quality_days_with_at_least_100_markets": quality_days,
        "quality_day_fraction": float(quality_days / len(all_days)),
        "coverage_gate_passed": coverage_gate,
        "primary": primary_result,
        "secondary": secondary_result,
        "benchmark_log_loss": {
            name: float(frame[f"ll_{name}"].mean())
            for name in ("p_market", "p_A_frozen", "p_C_market_only", "p_D_market_plus_btc")
        },
        "secondary_brier_mean": {
            name: float(frame[f"brier_{name}"].mean())
            for name in ("p_market", "p_A_frozen", "p_C_market_only", "p_D_market_plus_btc")
        },
        "secondary_calibration_diagnostics": calibration,
        "primary_verdict": prospective._primary_verdict(
            primary_result["ci_95_two_sided"], coverage_gate,
        ),
        "rule": "negative D-minus-raw-market log loss favors D; D-minus-C is secondary",
    }


def _write_reports(result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "final_results.json"
    md_path = OUTPUT_DIR / "final_results.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    lines = [
        "# Prospective BTC-plus-market test result",
        "",
        f"Status: **{result['status']}**",
        f"First scheduled market: `{result.get('first_market_start_utc')}`",
        f"Evaluation deadline: `{result.get('evaluation_deadline_utc')}`",
        f"Official outcome settlement deadline: `{result.get('official_outcome_settlement_deadline_utc')}`",
        "",
        "## Operational completeness",
        "",
        f"- Attempt coverage: {result['operational']['attempt_coverage']}",
        f"- Scheduled markets: {result['operational']['scheduled_market_count']}",
        f"- Recorded attempts: {result['operational']['append_only_attempt_count']}",
        f"- Late or missing attempts: {result['operational']['late_or_missing_attempts']}",
        f"- Official outcomes available by settlement deadline: {result['operational']['official_outcomes_available_by_settlement_deadline']}",
        f"- T-60 to forecast readiness (n, median, p95 seconds): {result['operational']['latency_seconds']['t60_deadline_to_quality_inputs_ready_seconds']['count']}, {result['operational']['latency_seconds']['t60_deadline_to_quality_inputs_ready_seconds']['median_seconds']}, {result['operational']['latency_seconds']['t60_deadline_to_quality_inputs_ready_seconds']['p95_seconds']}",
        f"- T-60 to BTC candle receipt (n, median, p95 seconds): {result['operational']['latency_seconds']['t60_deadline_to_btc_candle_receipt_seconds']['count']}, {result['operational']['latency_seconds']['t60_deadline_to_btc_candle_receipt_seconds']['median_seconds']}, {result['operational']['latency_seconds']['t60_deadline_to_btc_candle_receipt_seconds']['p95_seconds']}",
        f"- Feature construction (median, p95 seconds): {result['operational']['latency_seconds']['feature_construction_seconds']['median_seconds']}, {result['operational']['latency_seconds']['feature_construction_seconds']['p95_seconds']}",
        f"- Prediction plus calibration (median, p95 seconds): {result['operational']['latency_seconds']['prediction_and_calibration_seconds']['median_seconds']}, {result['operational']['latency_seconds']['prediction_and_calibration_seconds']['p95_seconds']}",
        f"- BTC candle sources: `{result['operational']['btc_candle_source_counts']}`; attempts with a disconnected feed: {result['operational']['attempts_with_disconnected_btc_feed']}",
        "",
    ]
    if result.get("quality"):
        quality = result["quality"]
        lines.extend([
            "## Forecast quality",
            "",
            f"- Common market count: {quality['common_quality_market_count']}",
            f"- Covered days with at least 100 markets: {quality['quality_days_with_at_least_100_markets']}",
            f"- Quality exclusions by reason: `{quality['quality_exclusion_counts']}`",
            "",
        ])
        if quality.get("primary"):
            lines.extend([
                f"- Primary verdict: **{quality['primary_verdict']}**",
                f"- Primary D minus raw market mean log-loss difference: {quality['primary']['mean_paired_difference']:.8f}",
                f"- Primary 95% interval: `{quality['primary']['ci_95_two_sided']}`",
                f"- Secondary D minus C mean difference: {quality['secondary']['mean_paired_difference']:.8f}",
                f"- Secondary 95% interval: `{quality['secondary']['ci_95_two_sided']}`",
                f"- Mean Brier scores: `{quality['secondary_brier_mean']}`",
                f"- Fixed-bin calibration diagnostics: `{quality['secondary_calibration_diagnostics']}`",
                "",
            ])
        else:
            lines.extend(["No scoreable common quality markets were available by the settlement deadline.", ""])
    if result.get("economics"):
        lines.extend([
            "## Simulated economics (separate from forecast quality)",
            "",
            f"- Fixed $5 realized net PnL: {result['economics']['fixed_5_usd']['realized_net_pnl_usd']:.4f} USD",
            f"- Fixed $5 priceable T-59 fills: {result['economics']['fixed_5_usd']['priceable_t59_fills']}",
            f"- 5% free cash simulated realized net PnL: {result['economics']['five_percent_free_cash']['realized_net_pnl_usd']:.4f} USD",
            "- Actual fills and fees remain unknown; all economics are simulated from recorded public books.",
            "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Prospective collector database not found: {DB_PATH}")
    connection = sqlite3.connect(DB_PATH)
    session = _read_session(connection)
    attempts = _read_attempts(connection)
    outcomes = _read_outcomes(connection)
    first_market = session["first_market_start_utc"]
    market_deadline = session["evaluation_deadline_utc"]
    if not first_market or not market_deadline:
        raise RuntimeError("Prospective observation window has not started")
    scheduled = _scheduled_markets(first_market, market_deadline)
    outcome_deadline = pd.to_datetime(market_deadline, utc=True) + pd.Timedelta(
        hours=int(prospective.PROTOCOL["quality_test"]["official_outcome_settlement_grace_hours"])
    )
    scheduled_slugs = {f"{prospective.MARKET_PREFIX}-{int(stamp.timestamp())}" for stamp in scheduled}
    attempts = [x for x in attempts if x["market_slug"] in scheduled_slugs]
    outcomes = {slug: item for slug, item in outcomes.items() if slug in scheduled_slugs}
    operational = _operational_summary(connection, attempts, outcomes, scheduled, outcome_deadline)
    now = pd.Timestamp.now(tz="UTC")
    collection_status = _collection_status(now, market_deadline, outcome_deadline)
    if collection_status == "collection_in_progress":
        result = {
            "status": "collection_in_progress",
            "first_market_start_utc": first_market,
            "evaluation_deadline_utc": market_deadline,
            "official_outcome_settlement_deadline_utc": outcome_deadline.isoformat(),
            "observed_at_utc": now.isoformat(),
            "protocol_sha256": session["protocol_sha256"],
            "artifact_sha256": session["artifact_sha256"],
            "operational": operational,
            "quality": None,
            "economics": None,
            "note": "Significance and forecast quality are intentionally withheld until the fixed deadline.",
        }
    elif collection_status == "settlement_collection_in_progress":
        result = {
            "status": "settlement_collection_in_progress",
            "first_market_start_utc": first_market,
            "evaluation_deadline_utc": market_deadline,
            "official_outcome_settlement_deadline_utc": outcome_deadline.isoformat(),
            "observed_at_utc": now.isoformat(),
            "protocol_sha256": session["protocol_sha256"],
            "artifact_sha256": session["artifact_sha256"],
            "operational": operational,
            "quality": None,
            "economics": None,
            "note": "The fixed market window has ended. Quality and economics remain withheld until the fixed official-outcome settlement grace ends.",
        }
    else:
        quality = _score(attempts, outcomes, scheduled, outcome_deadline, prospective.PROTOCOL)
        coverage_passed = (
            (operational["attempt_coverage"] or 0.0) >= prospective.PROTOCOL["quality_test"]["minimum_attempt_coverage"]
            and quality["quality_days_with_at_least_100_markets"] >= prospective.PROTOCOL["quality_test"]["minimum_quality_days"]
            and quality["common_quality_market_count"] >= prospective.PROTOCOL["quality_test"]["minimum_common_quality_markets"]
        )
        quality["coverage_gate_passed"] = bool(coverage_passed)
        if quality.get("primary"):
            quality["primary_verdict"] = prospective._primary_verdict(
                quality["primary"]["ci_95_two_sided"], coverage_passed
            )
        economics = _economic_summary(attempts, outcomes, outcome_deadline)
        result = {
            "status": "complete" if coverage_passed else "inconclusive_insufficient_coverage",
            "first_market_start_utc": first_market,
            "evaluation_deadline_utc": market_deadline,
            "official_outcome_settlement_deadline_utc": outcome_deadline.isoformat(),
            "observed_at_utc": now.isoformat(),
            "protocol_sha256": session["protocol_sha256"],
            "artifact_sha256": session["artifact_sha256"],
            "operational": operational,
            "quality": quality,
            "economics": economics,
        }
    connection.close()
    _write_reports(result)
    print(f"Prospective result report written: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
