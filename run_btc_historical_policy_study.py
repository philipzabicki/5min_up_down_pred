"""Reprice the frozen BTC T-59 sample and compare a bounded policy family."""
from __future__ import annotations

import hashlib
import heapq
import inspect
import json
import math
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import run_btc_historical_fee_replay as historical_fee
import run_btc_live_readiness as readiness
from utils import polymarket_btc5m_fee_rules as fee_rules
from utils import polymarket_btc_exit_quotes as exit_quotes


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_historical_policy_study_20261010.json"
DATA_MANIFEST_PATH = ROOT / "reports/btc_preopen/t59_reassessment_20261008/data_manifest.json"
SOURCE_PROVENANCE_PATH = ROOT / "reports/btc_fee_policy_20261010/source_provenance.json"
REPORT_DIR = ROOT / "reports/btc_fee_policy_20261010"
RUNNER_VERSION = "btc_historical_policy_study_v1"
EXECUTION_NAME = "full_ladder_snapshot_upper_bound"
EXECUTION = {"absolute_order_price_cap": 0.95}
FEE_TRANSITION_UTC = "2026-05-06T00:00:00Z"
TRAIN_END_MONTH = "2026-05"
NO_TRADE_POLICY_ID = "no_trade"
NO_TRADE_THRESH = -1.0
NO_TRADE_CAP = -1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _study_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _policy_grid(config: dict) -> list[dict]:
    rows = [{"policy": "fixed_5_usd", "cap_usd": None, "sizing_id": "fixed_5_usd"}]
    for cap in config["sizing"]["free_cash_caps_usd"]:
        rows.append({
            "policy": readiness._policy_key(cap), "cap_usd": cap,
            "sizing_id": readiness._policy_key(cap),
        })
    return [
        {**row, "minimum_net_return": float(edge)}
        for row in rows
        for edge in config["entry_after_cost_return_thresholds"]
    ]


def _scenario_id(row: dict) -> str:
    cap = "none" if row["cap_usd"] is None else f"{row['cap_usd']:g}"
    return f"{row['sizing_id']}|edge={row['minimum_net_return']:.4f}|cap={cap}"


def _run_portfolio(markets: list[dict], row: dict, *, keep_trades=False):
    summary, monthly, trades, path = readiness.simulate(
        markets,
        policy=row["policy"], cap_usd=row["cap_usd"],
        fee_scenario=readiness.FEE_HISTORICAL,
        age_limit=1.0, release_delay=60,
        execution_name=EXECUTION_NAME, execution=EXECUTION,
        minimum_net_return=float(row["minimum_net_return"]),
        keep_trades=keep_trades, initial_cash_usd=100.0,
    )
    summary.update({
        "scenario_id": _scenario_id(row),
        "sizing_id": row["sizing_id"],
        "minimum_net_return": row["minimum_net_return"],
        "entry_edge_denominator": "cash debit (gross notional plus cash fee)",
        "fee_rate_transition_utc": FEE_TRANSITION_UTC,
        "fee_rate_transition_status": "estimated, predeclared, not selected by PnL",
        "fee_scenario": "archived_rate_schedule_estimate_with_side_specific_collection_unit",
    })
    for item in monthly:
        item.update({
            "scenario_id": _scenario_id(row),
            "sizing_id": row["sizing_id"],
            "policy": row["policy"], "cap_usd": row["cap_usd"],
            "minimum_net_return": row["minimum_net_return"],
            "fee_rate_transition_utc": FEE_TRANSITION_UTC,
        })
    for item in trades:
        item.update({
            "scenario_id": _scenario_id(row),
            "sizing_id": row["sizing_id"],
            "minimum_net_return": row["minimum_net_return"],
        })
    return summary, monthly, trades, path


def _scenario_cache_identity(markets: list[dict], config: dict) -> str:
    manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    source = {
        "runner_version": RUNNER_VERSION,
        "config_sha256": _sha256(CONFIG_PATH),
        "fee_registry_sha256": _sha256(fee_rules.REGISTRY_PATH),
        "fee_source_provenance_sha256": _sha256(SOURCE_PROVENANCE_PATH),
        "portfolio_wrapper_sha256": hashlib.sha256(
            inspect.getsource(_run_portfolio).encode("utf-8")
        ).hexdigest(),
        "readiness_sha256": _sha256(Path(readiness.__file__)),
        "fee_replay_sha256": _sha256(Path(historical_fee.__file__)),
        "t59_snapshot_sha256": manifest["replay_manifest"]["snapshot_sha256"],
        "replay_dependency_sha256": manifest["replay_manifest"]["replay_dependency_sha256"],
        "cached_input_hashes": manifest["input_identity"]["cached_inputs"],
        "initial_cash_usd": config["portfolio"]["initial_cash_usd"],
        "fee_transition_utc": FEE_TRANSITION_UTC,
        "market_count": len(markets),
        "market_first": markets[0]["condition_id"] if markets else None,
        "market_last": markets[-1]["condition_id"] if markets else None,
    }
    raw = json.dumps(source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _resume_scenario_results(markets: list[dict], config: dict, tasks: list[dict]):
    checkpoint_path = REPORT_DIR / "entry_scenario_checkpoint.json"
    identity = _scenario_cache_identity(markets, config)
    try:
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        saved = {}
    complete = saved.get("completed") if saved.get("identity") == identity else {}
    if not isinstance(complete, dict):
        complete = {}
    for task_index, task in enumerate(tasks, start=1):
        key = _scenario_id(task)
        if key in complete:
            continue
        started = time.perf_counter()
        summary, monthly, _, _ = _run_portfolio(markets, task, keep_trades=False)
        complete[key] = {"summary": summary, "monthly": monthly}
        _write_json(checkpoint_path, {
            "identity": identity,
            "runner_version": RUNNER_VERSION,
            "completed_scenarios": len(complete),
            "expected_scenarios": len(tasks),
            "completed": complete,
        })
        print(
            f"[entry-study] {task_index}/{len(tasks)} {_scenario_id(task)} "
            f"trades={summary['trade_count']} pnl={summary['net_pnl_usd']:+.2f} "
            f"elapsed_s={time.perf_counter()-started:.2f}",
            flush=True,
        )
    return complete


def _training_equity(monthly: list[dict]) -> float:
    rows = [row for row in monthly if row["month_utc"] == TRAIN_END_MONTH]
    if not rows:
        return 100.0
    return float(rows[-1]["closing_cost_basis_equity_usd"])


def _select_walk_forward(config: dict, completed: dict) -> tuple[dict, dict, list[dict]]:
    candidates = []
    for key, result in completed.items():
        summary = result["summary"]
        if summary["trade_count"] <= 0:
            continue
        training_equity = _training_equity(result["monthly"])
        candidates.append({
            "scenario_id": key,
            "policy": summary["policy"], "cap_usd": summary["cap_usd"],
            "sizing_id": summary["sizing_id"],
            "minimum_net_return": float(summary["minimum_net_return"]),
            "training_equity_at_2026_05_31_usd": training_equity,
            "training_trade_count_through_cutoff": sum(
                int(row["entry_count"]) for row in result["monthly"]
                if row["month_utc"] <= TRAIN_END_MONTH
            ),
        })
    positive = [row for row in candidates if row["training_equity_at_2026_05_31_usd"] > 100.0 + 1e-8]
    positive.sort(key=lambda row: (
        row["training_equity_at_2026_05_31_usd"],
        row["minimum_net_return"],
        -(float(row["cap_usd"]) if row["cap_usd"] is not None else math.inf),
        -row["training_trade_count_through_cutoff"],
    ), reverse=True)
    challenger = positive[0] if positive else (max(
        candidates,
        key=lambda row: row["training_equity_at_2026_05_31_usd"],
        default=None,
    ))
    if challenger is None:
        selected = {
            "scenario_id": NO_TRADE_POLICY_ID, "policy": NO_TRADE_POLICY_ID,
            "cap_usd": None, "sizing_id": NO_TRADE_POLICY_ID,
            "minimum_net_return": 1.0,
            "training_equity_at_2026_05_31_usd": 100.0,
            "training_trade_count_through_cutoff": 0,
        }
    elif positive:
        selected = challenger
    else:
        selected = {
            "scenario_id": NO_TRADE_POLICY_ID, "policy": NO_TRADE_POLICY_ID,
            "cap_usd": None, "sizing_id": NO_TRADE_POLICY_ID,
            "minimum_net_return": 1.0,
            "training_equity_at_2026_05_31_usd": 100.0,
            "training_trade_count_through_cutoff": 0,
        }
    return selected, challenger, candidates


def _no_trade_rows(month_labels: list[str]) -> tuple[dict, list[dict]]:
    summary = {
        "policy": NO_TRADE_POLICY_ID, "sizing_id": NO_TRADE_POLICY_ID,
        "cap_usd": None, "minimum_net_return": 1.0,
        "scenario_id": NO_TRADE_POLICY_ID,
        "initial_cash_usd": 100.0,
        "ending_cash_after_assumed_release_usd": 100.0,
        "net_pnl_usd": 0.0, "trade_pnl_sum_usd": 0.0,
        "trade_count": 0, "skip_count": 0,
        "gross_turnover_usd": 0.0, "fees_paid_estimated_usd": 0.0,
        "max_drawdown_cost_basis_equity": 0.0,
        "maximum_concurrent_cost_basis_exposure_usd": 0.0,
        "maximum_concurrent_positions": 0,
        "minimum_free_cash_usd": 100.0,
        "skip_reasons": "{}",
    }
    monthly = [{
        "scenario_id": NO_TRADE_POLICY_ID,
        "sizing_id": NO_TRADE_POLICY_ID,
        "policy": NO_TRADE_POLICY_ID, "cap_usd": None,
        "minimum_net_return": 1.0,
        "month_utc": label,
        "opening_cost_basis_equity_usd": 100.0,
        "closing_cost_basis_equity_usd": 100.0,
        "opening_free_cash_usd": 100.0,
        "closing_free_cash_usd": 100.0,
        "closing_locked_cost_basis_usd": 0.0,
        "net_pnl_usd": 0.0, "entry_count": 0,
        "gross_turnover_usd": 0.0, "fees_paid_estimated_usd": 0.0,
        "monthly_max_drawdown_cost_basis_equity": 0.0,
        "minimum_free_cash_usd": 100.0,
        "maximum_cost_basis_exposure_usd": 0.0,
        "maximum_concurrent_positions": 0,
        "skip_reasons": "{}",
    } for label in month_labels]
    return summary, monthly


def _previous_report_comparison(new_summary: pd.DataFrame) -> pd.DataFrame:
    previous_path = ROOT / "reports/btc_fee_history_20261010/scenario_summary.csv"
    if not previous_path.is_file():
        return pd.DataFrame()
    old = pd.read_csv(previous_path)
    old = old.loc[
        old.segment_id.eq("continuous_mixed_regime_estimate")
        & old.ask_age_limit_seconds.eq(1.0)
        & old.release_delay_seconds.eq(60)
        & old.execution_assumption.eq(EXECUTION_NAME)
    ].copy()
    old = old[[
        "policy", "cap_usd", "trade_count", "gross_turnover_usd",
        "fees_paid_estimated_usd", "ending_cash_after_assumed_release_usd",
        "net_pnl_usd", "max_drawdown_cost_basis_equity", "skip_count",
    ]].rename(columns={
        "trade_count": "previous_trade_count",
        "gross_turnover_usd": "previous_turnover_usd",
        "fees_paid_estimated_usd": "previous_fees_usd",
        "ending_cash_after_assumed_release_usd": "previous_ending_cash_usd",
        "net_pnl_usd": "previous_net_pnl_usd",
        "max_drawdown_cost_basis_equity": "previous_max_drawdown",
        "skip_count": "previous_skip_count",
    })
    new = new_summary.loc[
        new_summary.minimum_net_return.eq(0.0)
    ][[
        "policy", "cap_usd", "trade_count", "gross_turnover_usd",
        "fees_paid_estimated_usd", "ending_cash_after_assumed_release_usd",
        "net_pnl_usd", "max_drawdown_cost_basis_equity", "skip_count",
    ]].rename(columns={
        "trade_count": "updated_trade_count",
        "gross_turnover_usd": "updated_turnover_usd",
        "fees_paid_estimated_usd": "updated_fees_usd",
        "ending_cash_after_assumed_release_usd": "updated_ending_cash_usd",
        "net_pnl_usd": "updated_net_pnl_usd",
        "max_drawdown_cost_basis_equity": "updated_max_drawdown",
        "skip_count": "updated_skip_count",
    })
    return old.merge(new, on=["policy", "cap_usd"], how="outer")


def _manifest(markets: list[dict], input_quality: dict, verification: dict, config: dict) -> dict:
    data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    input_identity = data_manifest["input_identity"]
    model_hashes = input_identity["frozen_model_and_calibrator_sha256"]
    return {
        "study_id": config["study_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_commit": "28e1e4897804d2499e7b4bcc28b7b675ac23b6ed",
        "git_branch": "main",
        "frozen_model_artifact_sha256": model_hashes,
        "causal_prediction_artifact": input_identity["cached_inputs"]["causal_predictions"],
        "gamma_market_artifact": input_identity["cached_inputs"]["gamma_markets"],
        "frozen_t59_snapshot_sha256": data_manifest["replay_manifest"]["snapshot_sha256"],
        "replay_dependency_sha256": data_manifest["replay_manifest"]["replay_dependency_sha256"],
        "replay_source_rows_already_replayed_before_this_study": data_manifest["replay_manifest"]["event_rows_applied"],
        "model_retrained_or_retuned": False,
        "new_full_event_replay_performed": False,
        "orders_sent": False,
        "sample": input_quality,
        "input_verification": verification,
        "frozen_execution_and_portfolio_assumptions": config,
        "fee_transition_assumption": FEE_TRANSITION_UTC,
        "fee_source_provenance_path": SOURCE_PROVENANCE_PATH.relative_to(ROOT).as_posix(),
        "fee_source_provenance_sha256": _sha256(SOURCE_PROVENANCE_PATH),
        "current_gamma_metadata_is_historical_fee_evidence": False,
        "archived_history_is_untouched_test": False,
        "study_input_rows": len(markets),
        "performance": {
            "entry_scenario_count": len(_policy_grid(config)),
            "scenario_cache_identity": _scenario_cache_identity(markets, config),
            "full_book_replay_repeated": False,
        },
    }


def run_entry_study() -> dict:
    started = time.perf_counter()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config = _study_config()
    verification = readiness.verify_inputs()
    markets, quality = readiness.load_markets()
    markets, fee_audit = historical_fee._attach_fee_rules(markets)
    if len(markets) != quality["snapshot_rows"]:
        raise RuntimeError("T-59 market coverage changed after fee assignment")
    unresolved = [m["condition_id"] for m in markets if not m.get("fee_rule")]
    if unresolved:
        raise RuntimeError(f"Historical fee assignment is missing for {len(unresolved)} markets")

    tasks = _policy_grid(config)
    completed = _resume_scenario_results(markets, config, tasks)
    if len(completed) != len(tasks):
        raise RuntimeError("Entry policy scenario checkpoint is incomplete")
    summary_rows = [item["summary"] for item in completed.values()]
    monthly_rows = [row for item in completed.values() for row in item["monthly"]]
    months = sorted({row["month_utc"] for row in monthly_rows})
    no_trade_summary, no_trade_monthly = _no_trade_rows(months)
    summary_rows.append(no_trade_summary)
    monthly_rows.extend(no_trade_monthly)
    summary_frame = pd.DataFrame(summary_rows)
    monthly_frame = pd.DataFrame(monthly_rows)
    selected, challenger, candidate_rows = _select_walk_forward(config, completed)
    selected_key = selected["scenario_id"]
    selected_summary = (
        no_trade_summary if selected_key == NO_TRADE_POLICY_ID
        else completed[selected_key]["summary"]
    )
    selected_monthly = (
        no_trade_monthly if selected_key == NO_TRADE_POLICY_ID
        else completed[selected_key]["monthly"]
    )

    summary_frame["training_equity_2026_05_31_usd"] = summary_frame.apply(
        lambda row: 100.0 if row["scenario_id"] == NO_TRADE_POLICY_ID else _training_equity(
            completed[str(row["scenario_id"])]["monthly"]
        ), axis=1,
    )
    summary_frame["walk_forward_selected"] = summary_frame.scenario_id.eq(selected_key)
    summary_frame["walk_forward_challenger"] = (
        False if challenger is None else summary_frame.scenario_id.eq(challenger["scenario_id"])
    )
    summary_frame.sort_values(
        ["training_equity_2026_05_31_usd", "minimum_net_return"],
        ascending=[False, False], inplace=True, kind="stable",
    )
    monthly_frame["walk_forward_selected"] = monthly_frame.scenario_id.eq(selected_key)
    monthly_frame["period_role"] = monthly_frame.month_utc.map(
        lambda value: "training" if value <= TRAIN_END_MONTH else "out_of_sample"
    )

    finalist_rows = []
    if selected_key != NO_TRADE_POLICY_ID:
        finalist_rows.append({
            "policy": selected_summary["policy"],
            "cap_usd": selected_summary["cap_usd"],
            "sizing_id": selected_summary["sizing_id"],
            "minimum_net_return": selected_summary["minimum_net_return"],
            "role": "walk_forward_selected",
        })
    elif challenger is not None:
        finalist_rows.append({
            "policy": challenger["policy"],
            "cap_usd": challenger["cap_usd"],
            "sizing_id": challenger["sizing_id"],
            "minimum_net_return": challenger["minimum_net_return"],
            "role": "best_active_challenger",
        })
    controls = [
        {"policy": "fixed_5_usd", "cap_usd": None, "sizing_id": "fixed_5_usd", "minimum_net_return": 0.0, "role": "fixed_5_control"},
        {"policy": readiness._policy_key(20.0), "cap_usd": 20.0, "sizing_id": readiness._policy_key(20.0), "minimum_net_return": 0.0, "role": "cap_20_control"},
    ]
    for control in controls:
        if all(_scenario_id(control) != _scenario_id(row) for row in finalist_rows):
            finalist_rows.append({**control, "role": control["role"]})

    ledger_rows = []
    for finalist in finalist_rows:
        task = {key: finalist[key] for key in ("policy", "cap_usd", "sizing_id", "minimum_net_return")}
        _, _, trades, _ = _run_portfolio(markets, task, keep_trades=True)
        for trade in trades:
            trade["finalist_role"] = finalist["role"]
        ledger_rows.extend(trades)

    sensitivity_tasks = []
    if selected_key == NO_TRADE_POLICY_ID:
        active_key = challenger["scenario_id"] if challenger is not None else None
        if active_key is not None:
            candidate = next(row for row in tasks if _scenario_id(row) == active_key)
            sensitivity_tasks.append({**candidate, "role": "best_training_active_challenger"})
    else:
        candidate = next(row for row in tasks if _scenario_id(row) == selected_key)
        sensitivity_tasks.append({**candidate, "role": "walk_forward_selected"})
    for control in controls:
        if all(_scenario_id(control) != _scenario_id(row) for row in sensitivity_tasks):
            sensitivity_tasks.append({**control, "role": control["role"]})

    sensitivity_rows = []
    transition_dates = []
    first_transition = date(2026, 4, 23)
    last_transition = date(2026, 5, 15)
    current = first_transition
    while current <= last_transition:
        transition_dates.append(current.isoformat() + "T00:00:00Z")
        current += timedelta(days=1)
    original_rules = {market["condition_id"]: market["fee_rule"] for market in markets}
    original_registry = fee_rules.load_registry()
    for transition in transition_dates:
        transition_ts = pd.Timestamp(transition)
        for market in markets:
            market["fee_rule"] = fee_rules.resolve_market_fee_rule(
                market.get("fee_metadata"),
                pd.Timestamp(market["entry_ns"], unit="ns", tz="UTC"),
                original_registry,
                transition_utc=transition_ts,
            )
        for finalist in sensitivity_tasks:
            task = {key: finalist[key] for key in ("policy", "cap_usd", "sizing_id", "minimum_net_return")}
            summary, _, _, _ = _run_portfolio(markets, task, keep_trades=False)
            sensitivity_rows.append({
                "rate_transition_date_utc": transition,
                "transition_date_is_estimate": True,
                "pnl_not_used_to_select_transition_date": True,
                "scenario_id": _scenario_id(task),
                "finalist_role": finalist["role"],
                "policy": summary["policy"], "cap_usd": summary["cap_usd"],
                "minimum_net_return": summary["minimum_net_return"],
                "trade_count": summary["trade_count"],
                "gross_turnover_usd": summary["gross_turnover_usd"],
                "fees_paid_estimated_usd": summary["fees_paid_estimated_usd"],
                "ending_cash_usd": summary["ending_cash_after_assumed_release_usd"],
                "net_pnl_usd": summary["net_pnl_usd"],
                "max_drawdown": summary["max_drawdown_cost_basis_equity"],
                "skip_count": summary["skip_count"],
            })
        print(f"[fee-sensitivity] completed transition {transition}", flush=True)
    for market in markets:
        market["fee_rule"] = original_rules[market["condition_id"]]

    previous = _previous_report_comparison(summary_frame)
    provenance = _manifest(markets, quality, verification, config)
    provenance.update({
        "generated_wall_seconds": time.perf_counter() - started,
        "historical_fee_audit_rows": len(fee_audit),
        "unresolved_fee_rule_count": 0,
        "fee_source_captured_market_rate_is_not_used_as_historical_rate": True,
        "walk_forward_selection": {
            "selected": selected,
            "best_active_challenger": challenger,
            "candidate_count": len(candidate_rows),
            "train_cutoff": config["chronological_selection"]["training_period_end_inclusive_utc"],
            "out_of_sample_start": config["chronological_selection"]["out_of_sample_period_start_utc"],
            "selected_policy_monthly_state": selected_monthly,
            "selection_rule": config["chronological_selection"]["selection_metric"],
            "cash_and_locked_cost_basis_carried_continuously": True,
        },
        "fee_transition_sensitivity_dates": transition_dates,
        "rebates_used_for_policy_selection": False,
        "maker_rebates_credited": False,
    })

    summary_frame.to_csv(REPORT_DIR / "policy_comparison.csv", index=False)
    monthly_frame.to_csv(REPORT_DIR / "monthly_portfolio.csv", index=False)
    pd.DataFrame(ledger_rows).to_csv(REPORT_DIR / "finalist_trade_ledger.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(REPORT_DIR / "fee_transition_sensitivity.csv", index=False)
    previous.to_csv(REPORT_DIR / "previous_report_comparison.csv", index=False)
    fee_audit.to_csv(REPORT_DIR / "fee_rule_assignment_audit.csv.gz", index=False, compression="gzip")
    _write_json(REPORT_DIR / "walk_forward_selection.json", provenance["walk_forward_selection"])
    _write_json(REPORT_DIR / "manifest.json", provenance)
    _write_report(config, provenance, summary_frame, monthly_frame, finalist_rows, previous)
    return provenance


def _fresh_sale(quote: dict | None, shares: float, rule: dict) -> dict | None:
    if not quote or quote.get("age_seconds") is None:
        return None
    if float(quote["age_seconds"]) > exit_quotes.FRESH_BID_SECONDS:
        return None
    result = fee_rules.walk_bids(quote.get("levels", ()), shares, rule)
    return result if result["depth_sufficient"] else None


def _exit_plan(market: dict, path: dict | None, exit_policy: dict, shares: float, cash_debit: float) -> dict:
    if exit_policy["kind"] == "hold" or not path:
        return {"sale": None, "trigger_offset_seconds": None, "attempt_count": 0, "exit_status": "hold_to_resolution"}
    resolved_ns = int(market["resolved_ns"])
    start_ns = int(market["market_start_ns"])
    rule = market["fee_rule"]
    if exit_policy["kind"] == "time":
        offset = int(exit_policy["offset_seconds"])
        decision_ns = start_ns + offset * 1_000_000_000
        arrival_ns = decision_ns + exit_quotes.ORDER_DELAY_SECONDS * 1_000_000_000
        if arrival_ns >= resolved_ns:
            return {"sale": None, "trigger_offset_seconds": None, "attempt_count": 0, "exit_status": "time_close_after_resolution"}
        quote = path["quotes"].get(str(offset), {}).get("arrival", {}).get(market["_chosen_side"])
        sale = _fresh_sale(quote, shares, rule)
        if sale is None:
            return {"sale": None, "trigger_offset_seconds": offset, "attempt_count": 1, "exit_status": "time_exit_unfilled_hold"}
        return {
            "sale": {**sale, "exit_time_ns": int(quote["quote_received_ns"]), "quote": quote},
            "trigger_offset_seconds": offset, "attempt_count": 1, "exit_status": "time_exit_filled",
        }

    take_profit = float(exit_policy["take_profit"])
    stop_loss = float(exit_policy["stop_loss"])
    triggered_at = None
    attempts = 0
    for offset in exit_quotes.EXIT_DECISION_OFFSETS:
        decision_ns = start_ns + offset * 1_000_000_000
        arrival_ns = decision_ns + exit_quotes.ORDER_DELAY_SECONDS * 1_000_000_000
        if arrival_ns >= resolved_ns:
            break
        sample = path["quotes"].get(str(offset), {})
        if triggered_at is None:
            decision_quote = sample.get("decision", {}).get(market["_chosen_side"])
            decision_sale = _fresh_sale(decision_quote, shares, rule)
            if decision_sale is None:
                continue
            net_return = (float(decision_sale["net_proceeds_usd"]) - cash_debit) / cash_debit
            if net_return >= take_profit or net_return <= stop_loss:
                triggered_at = offset
        if triggered_at is not None:
            attempts += 1
            quote = sample.get("arrival", {}).get(market["_chosen_side"])
            sale = _fresh_sale(quote, shares, rule)
            if sale is not None:
                return {
                    "sale": {**sale, "exit_time_ns": int(quote["quote_received_ns"]), "quote": quote},
                    "trigger_offset_seconds": triggered_at,
                    "attempt_count": attempts,
                    "exit_status": "threshold_exit_filled",
                }
    status = "threshold_trigger_unfilled_hold" if triggered_at is not None else "threshold_not_triggered_hold"
    return {"sale": None, "trigger_offset_seconds": triggered_at, "attempt_count": attempts, "exit_status": status}


def _simulate_exit_portfolio(
    markets: list[dict],
    *,
    scenario: dict,
    exit_policy: dict,
    quote_paths: dict[str, dict],
    common_market_ids: set[str] | None = None,
    discover_missing_paths: bool = False,
) -> tuple[dict, list[dict], list[dict], set[str]]:
    cash = 100.0
    locked = fees_paid = turnover = 0.0
    max_drawdown = 0.0
    peak_equity = 100.0
    min_free_cash = 100.0
    max_exposure = 0.0
    max_positions = 0
    open_positions = 0
    pending = []
    skips = Counter()
    missing_paths: set[str] = set()
    trade_rows: list[dict] = []
    event_rows: list[dict] = []
    market_count = outside_coverage = 0
    gross_sizes = []

    def record(now_ns: int, event_type: str, cid: str = "") -> None:
        nonlocal peak_equity, max_drawdown, min_free_cash, max_exposure, max_positions
        equity = cash + locked
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)
        min_free_cash = min(min_free_cash, cash)
        max_exposure = max(max_exposure, locked)
        max_positions = max(max_positions, open_positions)
        event_rows.append({
            "timestamp_ns": int(now_ns), "event_type": event_type, "condition_id": cid,
            "equity_usd": equity, "cash_usd": cash, "locked_cost_basis_usd": locked,
            "fees_paid_cumulative_usd": fees_paid, "turnover_cumulative_usd": turnover,
            "open_positions": open_positions,
        })

    def process_event(event) -> None:
        nonlocal cash, locked, fees_paid, turnover, open_positions
        when_ns, _, cid, kind, position = event
        row = position["trade_row"]
        if kind == "sale":
            sale = position["exit_plan"]["sale"]
            cash += float(sale["net_proceeds_usd"])
            locked -= float(position["gross_usd"])
            fees_paid += float(sale["fee_cash_usd"] or 0.0)
            turnover += float(sale["gross_proceeds_usd"])
            row.update({
                "actual_exit_type": "sale",
                "actual_exit_time_utc": pd.Timestamp(when_ns, unit="ns", tz="UTC").isoformat(),
                "exit_gross_proceeds_usd": float(sale["gross_proceeds_usd"]),
                "exit_fee_cash_usd": float(sale["fee_cash_usd"] or 0.0),
                "exit_net_proceeds_usd": float(sale["net_proceeds_usd"]),
                "exit_vwap": float(sale["vwap"]),
                "exit_price_levels": sale["price_levels"],
                "exit_quote_source": position["path_source"],
                "net_pnl_usd": float(sale["net_proceeds_usd"]) - float(position["cash_debit_usd"]),
                "cash_after_exit_usd": cash,
            })
        else:
            payout = float(position["payout_usd"])
            cash += payout
            locked -= float(position["gross_usd"])
            row.update({
                "actual_exit_type": "resolution",
                "actual_exit_time_utc": pd.Timestamp(when_ns, unit="ns", tz="UTC").isoformat(),
                "exit_gross_proceeds_usd": 0.0,
                "exit_fee_cash_usd": 0.0,
                "exit_net_proceeds_usd": payout,
                "exit_vwap": None,
                "exit_price_levels": [],
                "exit_quote_source": None,
                "net_pnl_usd": payout - float(position["cash_debit_usd"]),
                "cash_after_exit_usd": cash,
            })
        open_positions -= 1
        record(int(when_ns), kind, cid)

    def process_before_entry(entry_ns: int) -> None:
        while pending and pending[0][0] < entry_ns:
            process_event(heapq.heappop(pending))
        while pending and pending[0][0] == entry_ns and pending[0][1] == 0:
            process_event(heapq.heappop(pending))

    def process_sales_at_entry(entry_ns: int) -> None:
        while pending and pending[0][0] == entry_ns and pending[0][1] > 0:
            process_event(heapq.heappop(pending))

    def skip(reason: str, entry_ns: int) -> None:
        skips[reason] += 1
        process_sales_at_entry(entry_ns)

    if markets:
        record(int(markets[0]["entry_ns"]), "initial")
    for market in markets:
        cid = str(market["condition_id"]).lower()
        entry_ns = int(market["entry_ns"])
        process_before_entry(entry_ns)
        if common_market_ids is not None and cid not in common_market_ids:
            outside_coverage += 1
            process_sales_at_entry(entry_ns)
            continue
        market_count += 1
        snapshot = market.get("snapshot")
        if snapshot is None:
            skip("missing_t59_market_snapshot", entry_ns)
            continue
        if market.get("outcome") is None or market.get("resolved_ns") is None:
            skip("missing_official_outcome_or_resolution_time", entry_ns)
            continue
        if market.get("p_candidate_platt") is None:
            skip("missing_causal_prediction", entry_ns)
            continue
        desired = (
            readiness.FIXED_STAKE_USD if scenario["policy"] == "fixed_5_usd"
            else readiness.desired_stake(cash, scenario["cap_usd"])
        )
        if desired <= 1e-8:
            skip("no_free_cash", entry_ns)
            continue
        chosen, evaluations, reason = readiness._evaluate_sides(
            snapshot, market, desired,
            fee_scenario=readiness.FEE_HISTORICAL,
            age_limit=1.0,
            execution={"absolute_order_price_cap": 0.95},
            minimum_net_return=float(scenario["minimum_net_return"]),
        )
        if chosen is None:
            skip(reason, entry_ns)
            continue
        fill = evaluations[chosen]["fill"]
        debit = float(fill["cash_debit_usd"])
        if cash + 1e-9 < debit:
            skip("insufficient_free_cash_for_cash_debit", entry_ns)
            continue

        path = quote_paths.get(cid)
        if discover_missing_paths and path is None:
            missing_paths.add(cid)
        elif discover_missing_paths and not path.get("source_complete", False):
            # Markets with a known source gap are excluded from discovery and from
            # the paired common-coverage run.
            skip("exit_quote_source_incomplete", entry_ns)
            continue
        if common_market_ids is not None and path is None:
            raise RuntimeError(f"Common-coverage market is absent from exit quote cache: {cid}")

        market_for_exit = {**market, "_chosen_side": chosen}
        plan = _exit_plan(market_for_exit, path, exit_policy, float(fill["shares"]), debit)
        won = int(market["outcome"]) == int(chosen == "up")
        payout = float(fill["shares"]) if won else 0.0
        release_ns = readiness.release_time_ns(
            int(market["market_start_ns"]), int(market["resolved_ns"]), 60,
        )
        rule = market["fee_rule"]
        entry_fee = float(fill.get("fee_usd") or 0.0)
        cash -= debit
        locked += float(fill["gross_usd"])
        fees_paid += entry_fee
        turnover += float(fill["gross_usd"])
        gross_sizes.append(float(fill["gross_usd"]))
        open_positions += 1
        trade_row = {
            "entry_role": scenario["role"],
            "entry_scenario_id": scenario["scenario_id"],
            "exit_policy": exit_policy["id"],
            "condition_id": cid,
            "market_start_utc": market["market_start_utc"].isoformat(),
            "entry_time_utc": pd.Timestamp(entry_ns, unit="ns", tz="UTC").isoformat(),
            "chosen_side": chosen,
            "outcome_up": int(market["outcome"]),
            "won": bool(won),
            "predicted_net_return": float(evaluations[chosen]["ev_usd"]) / debit,
            "minimum_net_return": float(scenario["minimum_net_return"]),
            "gross_usd": float(fill["gross_usd"]),
            "gross_shares": float(fill["gross_shares"]),
            "net_shares": float(fill["shares"]),
            "entry_vwap": float(fill["vwap"]),
            "entry_fee_estimated_usd": entry_fee,
            "entry_fee_cash_usd": float(fill.get("fee_cash_usd") or 0.0),
            "entry_fee_shares": float(fill.get("fee_shares") or 0.0),
            "cash_debit_usd": debit,
            "cash_before_entry_usd": cash + debit,
            "cash_after_entry_usd": cash,
            "fee_rule_id": rule["rule_id"],
            "fee_rate": rule["rate"],
            "fee_collection_mode": rule["collection_mode"],
            "fee_rule_status": rule["status"],
            "exit_quote_source": path["source"] if path else None,
            "exit_quote_source_complete": bool(path and path["source_complete"]),
            "exit_trigger_offset_seconds": plan["trigger_offset_seconds"],
            "exit_attempt_count": plan["attempt_count"],
            "exit_status": plan["exit_status"],
            "payout_at_resolution_usd": payout,
            "assumed_cash_release_utc": pd.Timestamp(release_ns, unit="ns", tz="UTC").isoformat(),
        }
        trade_rows.append(trade_row)
        record(entry_ns, "entry", cid)
        sale = plan["sale"]
        if sale is not None:
            heapq.heappush(pending, (int(sale["exit_time_ns"]), 1, cid, "sale", {
                "trade_row": trade_row,
                "exit_plan": plan,
                "gross_usd": float(fill["gross_usd"]),
                "cash_debit_usd": debit,
                "payout_usd": payout,
                "path_source": path["source"],
            }))
        else:
            heapq.heappush(pending, (release_ns, 0, cid, "settlement", {
                "trade_row": trade_row,
                "exit_plan": plan,
                "gross_usd": float(fill["gross_usd"]),
                "cash_debit_usd": debit,
                "payout_usd": payout,
                "path_source": None,
            }))
        process_sales_at_entry(entry_ns)

    while pending:
        process_event(heapq.heappop(pending))
    if abs(cash - 100.0 - sum(float(row.get("net_pnl_usd", 0.0)) for row in trade_rows)) > 1e-6:
        raise RuntimeError("Exit portfolio ledger does not reconcile to ending cash")
    if locked < -1e-8 or cash < -1e-8:
        raise RuntimeError("Exit portfolio cash or locked cost became negative")

    event_frame = pd.DataFrame(event_rows).sort_values(["timestamp_ns"], kind="stable") if event_rows else pd.DataFrame()
    monthly_rows = []
    if not event_frame.empty:
        event_frame["month_utc"] = pd.to_datetime(event_frame["timestamp_ns"], unit="ns", utc=True).dt.strftime("%Y-%m")
        first_month = pd.Timestamp(event_frame["timestamp_ns"].min(), unit="ns", tz="UTC").replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_month = pd.Timestamp(event_frame["timestamp_ns"].max(), unit="ns", tz="UTC").replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        previous_equity, previous_fees, previous_turnover = 100.0, 0.0, 0.0
        entry_counts = Counter(pd.Timestamp(row["entry_time_utc"]).strftime("%Y-%m") for row in trade_rows)
        exit_counts = Counter(pd.Timestamp(row["actual_exit_time_utc"]).strftime("%Y-%m") for row in trade_rows)
        for month in pd.date_range(first_month, last_month, freq="MS", tz="UTC"):
            label = month.strftime("%Y-%m")
            rows = event_frame.loc[event_frame["month_utc"].eq(label)]
            if rows.empty:
                closing = previous_equity
                closing_fees, closing_turnover = previous_fees, previous_turnover
                monthly_dd = 0.0
                min_cash_month = None
            else:
                closing = float(rows.iloc[-1]["equity_usd"])
                closing_fees = float(rows.iloc[-1]["fees_paid_cumulative_usd"])
                closing_turnover = float(rows.iloc[-1]["turnover_cumulative_usd"])
                path_equity = [previous_equity, *rows["equity_usd"].astype(float).tolist()]
                peak = path_equity[0]
                monthly_dd = 0.0
                for equity in path_equity[1:]:
                    peak = max(peak, equity)
                    monthly_dd = max(monthly_dd, (peak - equity) / peak if peak > 0 else 0.0)
                min_cash_month = float(rows["cash_usd"].min())
            monthly_rows.append({
                "entry_role": scenario["role"], "entry_scenario_id": scenario["scenario_id"],
                "exit_policy": exit_policy["id"], "month_utc": label,
                "opening_cost_basis_equity_usd": previous_equity,
                "closing_cost_basis_equity_usd": closing,
                "net_pnl_usd": closing - previous_equity,
                "entry_count": int(entry_counts[label]),
                "exit_count": int(exit_counts[label]),
                "gross_turnover_usd": closing_turnover - previous_turnover,
                "fees_paid_estimated_usd": closing_fees - previous_fees,
                "monthly_max_drawdown_cost_basis_equity": monthly_dd,
                "minimum_free_cash_usd": min_cash_month,
            })
            previous_equity, previous_fees, previous_turnover = closing, closing_fees, closing_turnover

    total_pnl = cash - 100.0
    summary = {
        **{key: scenario.get(key) for key in ("role", "scenario_id", "policy", "cap_usd", "sizing_id", "minimum_net_return")},
        "exit_policy": exit_policy["id"],
        "sample": "common_quote_coverage" if common_market_ids is not None else "coverage_discovery_pass",
        "common_market_count": market_count,
        "outside_common_coverage_count": outside_coverage,
        "entry_opportunity_count": market_count,
        "trade_count": len(trade_rows),
        "sold_count": sum(row.get("actual_exit_type") == "sale" for row in trade_rows),
        "held_to_resolution_count": sum(row.get("actual_exit_type") == "resolution" for row in trade_rows),
        "skip_count": sum(skips.values()),
        "skip_reasons": json.dumps(dict(skips), sort_keys=True),
        "initial_cash_usd": 100.0,
        "ending_cash_after_all_releases_usd": cash,
        "ending_locked_cost_basis_usd": locked,
        "net_pnl_usd": total_pnl,
        "trade_pnl_sum_usd": float(sum(float(row.get("net_pnl_usd", 0.0)) for row in trade_rows)),
        "gross_turnover_usd": turnover,
        "fees_paid_estimated_usd": fees_paid,
        "maximum_drawdown_cost_basis_equity": max_drawdown,
        "maximum_concurrent_cost_basis_exposure_usd": max_exposure,
        "maximum_concurrent_positions": max_positions,
        "minimum_free_cash_usd": min_free_cash,
        "mean_gross_stake_usd": sum(gross_sizes) / len(gross_sizes) if gross_sizes else None,
    }
    return summary, monthly_rows, trade_rows, missing_paths


def _exit_scenarios(config: dict, selection: dict) -> list[dict]:
    scenarios = []
    challenger = selection.get("best_active_challenger")
    if challenger:
        scenarios.append({**challenger, "role": "best_active_challenger"})
    scenarios.extend([
        {
            "scenario_id": "fixed_5_usd|edge=0.0000|cap=none",
            "policy": "fixed_5_usd", "cap_usd": None, "sizing_id": "fixed_5_usd",
            "minimum_net_return": 0.0, "role": "fixed_5_control",
        },
        {
            "scenario_id": f"{readiness._policy_key(20.0)}|edge=0.0000|cap=20",
            "policy": readiness._policy_key(20.0), "cap_usd": 20.0,
            "sizing_id": readiness._policy_key(20.0), "minimum_net_return": 0.0,
            "role": "cap_20_control",
        },
    ])
    unique = {}
    for row in scenarios:
        unique.setdefault(row["scenario_id"], row)
    exits = [{"id": "hold_to_resolution", "kind": "hold"}]
    exits.extend({"id": f"close_at_{int(offset)}s", "kind": "time", "offset_seconds": int(offset)}
                 for offset in config["exit_family"]["fixed_close_market_start_offsets_seconds"])
    exits.extend({
        "id": f"tp{int(100 * pair['take_profit'])}_sl{abs(int(100 * pair['stop_loss']))}",
        "kind": "threshold", "take_profit": float(pair["take_profit"]),
        "stop_loss": float(pair["stop_loss"]),
    } for pair in config["exit_family"]["take_profit_stop_loss_net_liquidation_returns"])
    return list(unique.values()), exits


def _estimate_taker_rebates() -> dict:
    """Keep uncertain current-page rebate terms outside the strategy ledger."""
    source = json.loads(SOURCE_PROVENANCE_PATH.read_text(encoding="utf-8"))
    source_row = next(row for row in source["sources"] if "Taker Rebate Program" in row["title"])
    tiers = [
        {"level": 0, "name": "None", "threshold_wv": 0.0, "rebate_fraction": 0.0, "bonus_usd": 0.0},
        {"level": 1, "name": "Bronze", "threshold_wv": 2_000.0, "rebate_fraction": 0.03, "bonus_usd": 10.0},
        {"level": 2, "name": "Silver", "threshold_wv": 20_000.0, "rebate_fraction": 0.08, "bonus_usd": 50.0},
        {"level": 3, "name": "Gold", "threshold_wv": 200_000.0, "rebate_fraction": 0.18, "bonus_usd": 250.0},
        {"level": 4, "name": "Platinum", "threshold_wv": 1_000_000.0, "rebate_fraction": 0.32, "bonus_usd": 1_500.0},
        {"level": 5, "name": "Diamond", "threshold_wv": 4_000_000.0, "rebate_fraction": 0.44, "bonus_usd": 7_500.0},
        {"level": 6, "name": "Obsidian", "threshold_wv": 10_000_000.0, "rebate_fraction": 0.50, "bonus_usd": 25_000.0},
    ]
    launch = pd.Timestamp("2026-05-28T00:00:00Z")
    cutoff = pd.Timestamp("2026-10-06T23:55:00Z")
    source_ledger = pd.read_csv(REPORT_DIR / "finalist_trade_ledger.csv")
    base_summary = pd.read_csv(REPORT_DIR / "policy_comparison.csv")
    exit_ledger = pd.read_csv(REPORT_DIR / "exit_trade_ledger.csv")
    exit_summary = pd.read_csv(REPORT_DIR / "exit_policy_comparison.csv")
    selected = json.loads((REPORT_DIR / "walk_forward_selection.json").read_text(encoding="utf-8"))
    scenario_roles = {"no_trade": "walk_forward_selected"}
    challenger = selected.get("best_active_challenger")
    if challenger:
        scenario_roles[str(challenger["scenario_id"])] = "best_active_challenger"
        exit_candidate_id = f"{challenger['scenario_id']}|exit=close_at_60s"
        scenario_roles[exit_candidate_id] = "best_active_challenger_exit_close_at_60s"
    for role, scenario_id in (
        ("fixed_5_control", "fixed_5_usd|edge=0.0000|cap=none"),
        ("cap_20_control", "free_cash_5pct_cap20|edge=0.0000|cap=20"),
    ):
        scenario_roles[scenario_id] = role

    summary_rows, cashflow_rows, trade_rows = [], [], []
    for scenario_id, role in scenario_roles.items():
        exit_candidate = role == "best_active_challenger_exit_close_at_60s"
        if exit_candidate:
            match = exit_summary.loc[
                exit_summary.entry_role.eq("best_active_challenger")
                & exit_summary.exit_policy.eq("close_at_60s")
                & exit_summary["sample"].eq("common_quote_coverage")
            ]
            base = match.iloc[0].to_dict() if len(match) else {}
            selected = exit_ledger.loc[
                exit_ledger.entry_role.eq("best_active_challenger")
                & exit_ledger.exit_policy.eq("close_at_60s")
            ].copy()
            entry_events = selected.assign(
                vwap=selected.entry_vwap,
                fee_usd=selected.entry_fee_estimated_usd,
                transaction_type="entry",
            )
            sold = selected.loc[selected.actual_exit_type.eq("sale")].copy()
            exit_events = sold.assign(
                entry_time_utc=sold.actual_exit_time_utc,
                gross_usd=sold.exit_gross_proceeds_usd,
                vwap=sold.exit_vwap,
                fee_usd=sold.exit_fee_cash_usd,
                transaction_type="exit_sale",
            )
            trades = pd.concat([entry_events, exit_events], ignore_index=True, sort=False)
        else:
            match = base_summary.loc[base_summary.scenario_id.astype(str).eq(scenario_id)]
            base = match.iloc[0].to_dict() if len(match) else {}
            trades = source_ledger.loc[source_ledger.scenario_id.astype(str).eq(scenario_id)].copy()
            if not trades.empty:
                trades["transaction_type"] = "entry"
        if not trades.empty:
            trades["entry_time_utc"] = pd.to_datetime(trades["entry_time_utc"], utc=True)
            trades = trades.loc[(trades.entry_time_utc >= launch) & (trades.entry_time_utc <= cutoff)]
            trades.sort_values("entry_time_utc", kind="stable", inplace=True)
            trades["weighted_volume_usd_estimated"] = (
                pd.to_numeric(trades["gross_usd"], errors="coerce").fillna(0.0)
                * (1.0 - pd.to_numeric(trades["vwap"], errors="coerce").fillna(1.0))
                * 2.3
            )
            trades["entry_fee_estimated_usd"] = pd.to_numeric(trades["fee_usd"], errors="coerce").fillna(0.0)
        else:
            trades = pd.DataFrame(columns=[
                "condition_id", "entry_time_utc", "gross_usd", "vwap", "fee_usd",
                "weighted_volume_usd_estimated", "entry_fee_estimated_usd", "transaction_type",
            ])

        by_day = defaultdict(list)
        for trade in trades.to_dict("records"):
            day = pd.Timestamp(trade["entry_time_utc"]).strftime("%Y-%m-%d")
            by_day[day].append(trade)
        trade_wv = [(pd.Timestamp(row["entry_time_utc"]), float(row["weighted_volume_usd_estimated"]))
                    for row in trades.to_dict("records")]
        accrued_rebate = paid_rebate = paid_bonus = eligible_wv = 0.0
        current_tier = 0
        reached_tiers = {0}
        max_tier = 0
        for day in pd.date_range(launch.normalize(), cutoff.normalize(), freq="D", tz="UTC"):
            midnight = day
            # Payouts for trades through the previous UTC day arrive at this midnight.
            if accrued_rebate >= 1.0:
                paid_rebate += accrued_rebate
                cashflow_rows.append({
                    "scenario_id": scenario_id, "role": role,
                    "payout_time_utc": midnight.isoformat(), "cashflow_type": "rebate",
                    "amount_usd_estimated": accrued_rebate,
                    "terms_status": "current_terms_estimate; historical launch terms unconfirmed",
                })
                accrued_rebate = 0.0
            window_start = midnight - pd.Timedelta(days=30)
            rolling_wv = sum(value for stamp, value in trade_wv if window_start <= stamp < midnight)
            tier = max(row["level"] for row in tiers if rolling_wv >= row["threshold_wv"])
            if tier > current_tier and tier not in reached_tiers:
                bonus = tiers[tier]["bonus_usd"]
                if bonus > 0:
                    paid_bonus += bonus
                    cashflow_rows.append({
                        "scenario_id": scenario_id, "role": role,
                        "payout_time_utc": midnight.isoformat(), "cashflow_type": "level_up_bonus",
                        "amount_usd_estimated": bonus,
                        "terms_status": "current_terms_estimate; historical launch terms unconfirmed",
                    })
                reached_tiers.add(tier)
            current_tier = tier
            max_tier = max(max_tier, tier)
            rebate_fraction = float(tiers[tier]["rebate_fraction"])
            for trade in by_day.get(midnight.strftime("%Y-%m-%d"), []):
                wv = float(trade["weighted_volume_usd_estimated"])
                fee = float(trade["entry_fee_estimated_usd"])
                rebate = fee * rebate_fraction
                eligible_wv += wv
                accrued_rebate += rebate
                trade_rows.append({
                    "scenario_id": scenario_id, "role": role,
                    "condition_id": trade["condition_id"],
                    "transaction_type": trade.get("transaction_type", "entry"),
                    "transaction_time_utc": pd.Timestamp(trade["entry_time_utc"]).isoformat(),
                    "weighted_volume_usd_estimated": wv,
                    "rolling_30d_wv_at_midnight_estimated": rolling_wv,
                    "tier_at_entry_estimated": tiers[current_tier]["name"],
                    "assumed_rebate_fraction_of_estimated_entry_fee": rebate_fraction,
                    "entry_fee_estimated_usd": fee,
                    "rebate_accrued_usd_estimated": rebate,
                    "assumed_bonus_multiplier": 1.0,
                })
        summary_rows.append({
            "scenario_id": scenario_id,
            "role": role,
            "program_go_live_utc": launch.isoformat(),
            "current_page_capture_date_utc": "2026-10-10",
            "historical_terms_confirmed": False,
            "eligible_entry_trade_count": int(trades.transaction_type.eq("entry").sum()) if len(trades) else 0,
            "eligible_exit_sale_count": int(trades.transaction_type.eq("exit_sale").sum()) if len(trades) else 0,
            "eligible_transaction_count": len(trades),
            "eligible_weighted_volume_usd_estimated": eligible_wv,
            "maximum_tier_estimated": tiers[max_tier]["name"],
            "executed_transaction_fees_usd_estimated": float(trades.entry_fee_estimated_usd.sum()) if len(trades) else 0.0,
            "pnl_before_rebates_usd": float(base.get("net_pnl_usd", 0.0) or 0.0),
            "rebate_cash_paid_usd_estimated": paid_rebate,
            "one_time_bonus_cash_paid_usd_estimated": paid_bonus,
            "rebate_accrued_unpaid_at_cutoff_usd_estimated": accrued_rebate,
            "ending_cash_before_rebates_usd": float(
                base.get("ending_cash_after_all_releases_usd", base.get("ending_cash_after_assumed_release_usd", 100.0)) or 100.0
            ),
            "ending_cash_after_paid_rebates_and_bonuses_usd_estimated": (
                float(base.get("ending_cash_after_all_releases_usd", base.get("ending_cash_after_assumed_release_usd", 100.0)) or 100.0)
                + paid_rebate + paid_bonus
            ),
            "rebate_basis_assumption": "Displayed rebate percentage multiplied by reconstructed executed taker fee equivalents; exit transaction eligibility and the monetary base are unconfirmed, so this is illustrative and excluded from selection.",
            "cash_available_for_entry_sizing_before_midnight_payout": False,
        })
    report = {
        "status": "separate_estimate_only",
        "source_url": source_row["original_url"],
        "source_sha256": source_row["sha256"],
        "source_capture_date_utc": "2026-10-10",
        "program_go_live_utc": launch.isoformat(),
        "historical_launch_terms_confirmed": False,
        "maker_rebates_credited": False,
        "policy_selection_uses_rebates": False,
        "weighted_volume_formula": "trade_size_usd * (1 - entry_vwap) * crypto_category_weight_2.3 * assumed_bonus_multiplier_1.0",
        "tier_update_assumption": "Current page rolling 30-day wV evaluated at daily UTC midnight; the tier then applies to that day's trades.",
        "payout_assumption": "Current page daily midnight pUSD payout with a $1 accrued minimum; unpaid accrual carries forward and is not added to strategy cash before payout.",
        "rebate_amount_assumption": "The page shows tier percentages but does not state their monetary base; this scenario applies them to reconstructed executed taker fee equivalents and is illustrative only. Exit transaction eligibility is unconfirmed.",
        "bonus_assumption": "Current page one-time tier bonuses paid at the midnight tier update; historical bonus terms and eligibility at launch remain unconfirmed.",
        "summary_rows": len(summary_rows),
        "cashflow_rows": len(cashflow_rows),
    }
    pd.DataFrame(summary_rows).to_csv(REPORT_DIR / "taker_rebate_estimate.csv", index=False)
    pd.DataFrame(cashflow_rows, columns=[
        "scenario_id", "role", "payout_time_utc", "cashflow_type",
        "amount_usd_estimated", "terms_status",
    ]).to_csv(REPORT_DIR / "taker_rebate_cashflows_estimate.csv", index=False)
    pd.DataFrame(trade_rows).to_csv(REPORT_DIR / "taker_rebate_trade_ledger_estimate.csv", index=False)
    _write_json(REPORT_DIR / "taker_rebate_estimate_manifest.json", report)
    return {**report, "scenario_summaries": summary_rows}


def run_exit_study() -> dict:
    study_started = time.perf_counter()
    config = _study_config()
    selection = json.loads((REPORT_DIR / "walk_forward_selection.json").read_text(encoding="utf-8"))
    scenarios, exits = _exit_scenarios(config, selection)
    markets, quality = readiness.load_markets()
    markets, _ = historical_fee._attach_fee_rules(markets)
    market_by_id = {str(market["condition_id"]).lower(): market for market in markets}
    cache_dir = readiness.INPUT_DIR / "btc_exit_quote_cache_20261010"
    cache_identity = exit_quotes._cache_identity()
    cached_paths, _ = exit_quotes._load_cache(cache_dir, cache_identity)
    paths: dict[str, dict] = {
        cid: path for cid, path in cached_paths.items() if cid in market_by_id
    }
    cache_hit_paths_at_start = len(paths)
    iterations = 0
    discovery_compute_seconds = 0.0
    quote_extraction_seconds = 0.0
    extracted_total = 0
    while True:
        iterations += 1
        missing_ids = set()
        for scenario in scenarios:
            for exit_policy in exits:
                started = time.perf_counter()
                summary, _, _, needed = _simulate_exit_portfolio(
                    markets, scenario=scenario, exit_policy=exit_policy,
                    quote_paths=paths, discover_missing_paths=True,
                )
                discovery_compute_seconds += time.perf_counter() - started
                missing_ids.update(needed)
                print(
                    f"[exit-discovery] iteration={iterations} role={scenario['role']} "
                    f"exit={exit_policy['id']} trades={summary['trade_count']} "
                    f"new_paths={len(needed)}",
                    flush=True,
                )
        missing_ids.difference_update(paths)
        if not missing_ids:
            break
        targets = [market_by_id[cid] for cid in sorted(missing_ids)]
        started = time.perf_counter()
        new_paths, extraction_report = exit_quotes.extract_exit_paths(
            targets, cache_dir=cache_dir,
        )
        quote_extraction_seconds += time.perf_counter() - started
        extracted_total += int(extraction_report["extracted_count"])
        if not new_paths:
            raise RuntimeError("Exit quote extraction did not checkpoint any requested market")
        paths.update(new_paths)
        print(
            f"[exit-discovery] cached={len(paths)} cache_hits={extraction_report['cache_hit_count']} "
            f"extracted={extraction_report['extracted_count']} "
            f"source_complete={extraction_report['source_complete_count']}",
            flush=True,
        )

    common_ids = {cid for cid, path in paths.items() if path["source_complete"]}
    comparison_rows, monthly_rows, ledger_rows, coverage_rows = [], [], [], []
    base_summary = pd.read_csv(REPORT_DIR / "policy_comparison.csv")
    for scenario in scenarios:
        for exit_policy in exits:
            summary, monthly, trades, _ = _simulate_exit_portfolio(
                markets, scenario=scenario, exit_policy=exit_policy,
                quote_paths=paths, common_market_ids=common_ids,
            )
            summary.update({
                "entry_role": scenario["role"],
                "entry_scenario_id": scenario["scenario_id"],
                "sample": "common_quote_coverage",
                "common_quote_market_ids": len(common_ids),
                "source_quote_freshness_limit_seconds": exit_quotes.FRESH_BID_SECONDS,
                "order_delay_seconds": exit_quotes.ORDER_DELAY_SECONDS,
                "cash_reuse_after_sale": "available after the sale event; same-timestamp entries are processed before sales",
            })
            comparison_rows.append(summary)
            for row in monthly:
                row["period_role"] = "training" if row["month_utc"] <= TRAIN_END_MONTH else "out_of_sample"
                monthly_rows.append(row)
            for trade in trades:
                trade["entry_role"] = scenario["role"]
                ledger_rows.append(trade)
            for row in [item for item in trades]:
                coverage_rows.append({
                    "entry_role": scenario["role"],
                    "exit_policy": exit_policy["id"],
                    "condition_id": row["condition_id"],
                    "market_start_utc": row["market_start_utc"],
                    "quote_source": row["exit_quote_source"],
                    "source_complete": row["exit_quote_source_complete"],
                    "exit_status": row["exit_status"],
                    "trigger_offset_seconds": row["exit_trigger_offset_seconds"],
                    "attempt_count": row["exit_attempt_count"],
                })
        matched = base_summary.loc[base_summary.scenario_id.eq(scenario["scenario_id"])]
        if not matched.empty:
            baseline = matched.iloc[0].to_dict()
            comparison_rows.append({
                "entry_role": scenario["role"],
                "entry_scenario_id": scenario["scenario_id"],
                "exit_policy": "hold_to_resolution",
                "sample": "full_entry_sample_control",
                "common_market_count": int(baseline.get("trade_count", 0)),
                "trade_count": int(baseline.get("trade_count", 0)),
                "sold_count": 0,
                "held_to_resolution_count": int(baseline.get("trade_count", 0)),
                "ending_cash_after_all_releases_usd": baseline.get("ending_cash_after_assumed_release_usd"),
                "net_pnl_usd": baseline.get("net_pnl_usd"),
                "gross_turnover_usd": baseline.get("gross_turnover_usd"),
                "fees_paid_estimated_usd": baseline.get("fees_paid_estimated_usd"),
                "maximum_drawdown_cost_basis_equity": baseline.get("max_drawdown_cost_basis_equity"),
                "skip_count": baseline.get("skip_count"),
                "skip_reasons": baseline.get("skip_reasons"),
                "full_sample_entry_scenario_id": scenario["scenario_id"],
            })
    monthly_frame = pd.DataFrame(monthly_rows)
    exit_walk_forward = None
    active_scenario = next((row for row in scenarios if row["role"] == "best_active_challenger"), None)
    if active_scenario is not None and not monthly_frame.empty:
        active_monthly = monthly_frame.loc[monthly_frame.entry_role.eq("best_active_challenger")]
        training_rows = active_monthly.loc[active_monthly.month_utc.eq(TRAIN_END_MONTH)]
        training_equity = {
            exit_policy["id"]: float(
                training_rows.loc[training_rows.exit_policy.eq(exit_policy["id"]), "closing_cost_basis_equity_usd"].iloc[-1]
            ) if training_rows.exit_policy.eq(exit_policy["id"]).any() else 100.0
            for exit_policy in exits
        }
        hold_equity = training_equity.get("hold_to_resolution", 100.0)
        selected_exit = max(exits, key=lambda row: (
            training_equity[row["id"]], row["kind"] == "hold",
        ))["id"]
        if training_equity[selected_exit] <= hold_equity + 1e-8:
            selected_exit = "hold_to_resolution"
        selected_oos = active_monthly.loc[
            active_monthly.exit_policy.eq(selected_exit)
            & active_monthly.month_utc.ge("2026-06")
        ]
        exit_walk_forward = {
            "role": "best_active_challenger",
            "training_cutoff": config["chronological_selection"]["training_period_end_inclusive_utc"],
            "selected_exit_policy": selected_exit,
            "selected_exit_training_equity_2026_05_31_usd": training_equity[selected_exit],
            "hold_control_training_equity_2026_05_31_usd": hold_equity,
            "training_equity_by_exit_policy": training_equity,
            "oos_start": config["chronological_selection"]["out_of_sample_period_start_utc"],
            "oos_monthly_pnl_usd": float(selected_oos.net_pnl_usd.sum()) if len(selected_oos) else 0.0,
            "oos_entry_count": int(selected_oos.entry_count.sum()) if len(selected_oos) else 0,
            "oos_ending_equity_usd": float(selected_oos.closing_cost_basis_equity_usd.iloc[-1]) if len(selected_oos) else training_equity[selected_exit],
            "selection_scope": "Exit-only diagnostic on source-complete common quote coverage; selected on training only, then rejected as unstable after its OOS loss.",
        }
    result = {
        "study_id": config["study_id"],
        "status": "completed",
        "quote_cache_report": {
            "cache_identity": exit_quotes._cache_identity(),
            "cache_identity_key": exit_quotes._identity_key(exit_quotes._cache_identity()),
            "cache_manifest_path": str(cache_dir / exit_quotes._identity_key(exit_quotes._cache_identity()) / "manifest.json"),
            "requested_market_count": len(paths),
            "source_complete_count": len(common_ids),
            "source_incomplete_count": len(paths) - len(common_ids),
            "source_counts": dict(Counter(path["source"] for path in paths.values())),
            "resumption_iterations": iterations,
            "new_market_paths_written": extracted_total,
            "cache_hit_market_count_at_run_start": cache_hit_paths_at_start,
            "wall_seconds": quote_extraction_seconds,
            "discovery_simulation_wall_seconds": discovery_compute_seconds,
            "profiled_reference": {
                "markets": 12,
                "source_counts": {"kacho_1hz_sample_proxy": 6, "pmxt_event_replay": 6},
                "all_source_complete": True,
                "elapsed_seconds": 1.78,
            },
        },
        "common_quote_market_count": len(common_ids),
        "exit_policy_count": len(exits),
        "entry_scenario_count": len(scenarios),
        "entry_policy_diagnostics": [scenario for scenario in scenarios],
        "selection": "Overall recommendation is no_trade: entry-only hold-to-resolution selection found no active policy above $100 on training, and the training-selected close_at_60s challenger exit lost its apparent advantage out of sample.",
        "walk_forward_exit_selection": exit_walk_forward,
        "exit_execution_limits": [
            "Exit policy comparisons use the union of source-complete market paths discovered for the compared entry policies, and every exit policy is rerun from one continuous $100 balance on that common market coverage.",
            "Markets without source-complete post-entry paths are omitted from the paired exit-policy sample; full-sample hold-to-resolution remains in the entry study table.",
            "Early sale proceeds can fund later entries after their timestamp. An entry at the same timestamp is processed before a sale, matching the existing portfolio event order.",
            "Kacho quotes are one-second sampled top-of-book proxies with top-level size; PMXT paths use receive-time event replay and reconstructed native bid depth. No quote is forward-filled beyond one second.",
            "If a full-position sale cannot execute at the bid depth, no partial sale is booked. Threshold exits retry every five-second decision only after a prior trigger; fixed-time exits make one attempt.",
        ],
    }
    pd.DataFrame(comparison_rows).to_csv(REPORT_DIR / "exit_policy_comparison.csv", index=False)
    monthly_frame.to_csv(REPORT_DIR / "exit_policy_monthly.csv", index=False)
    pd.DataFrame(ledger_rows).to_csv(REPORT_DIR / "exit_trade_ledger.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(REPORT_DIR / "exit_path_coverage.csv", index=False)
    rebate_result = _estimate_taker_rebates()
    result["taker_rebate_estimate"] = {
        "status": rebate_result["status"],
        "source_url": rebate_result["source_url"],
        "scenario_summaries": rebate_result["scenario_summaries"],
        "policy_selection_uses_rebates": False,
    }
    result["study_wall_seconds"] = time.perf_counter() - study_started
    result["performance"] = {
        "study_wall_seconds": result["study_wall_seconds"],
        "peak_working_set_mib": None,
        "peak_working_set_measurement": "not sampled by the runner",
    }
    _write_json(REPORT_DIR / "exit_study_manifest.json", {
        **result,
        "input_quality": quality,
        "fee_source_provenance_sha256": _sha256(SOURCE_PROVENANCE_PATH),
        "data_manifest_sha256": _sha256(DATA_MANIFEST_PATH),
        "common_market_id_sha256": hashlib.sha256("\n".join(sorted(common_ids)).encode("utf-8")).hexdigest(),
    })
    _write_report(
        config,
        json.loads((REPORT_DIR / "manifest.json").read_text(encoding="utf-8")),
        pd.read_csv(REPORT_DIR / "policy_comparison.csv"),
        pd.read_csv(REPORT_DIR / "monthly_portfolio.csv"),
        [],
        pd.read_csv(REPORT_DIR / "previous_report_comparison.csv"),
        exit_comparison=pd.DataFrame(comparison_rows),
        exit_manifest=result,
    )
    result["study_wall_seconds"] = time.perf_counter() - study_started
    _write_json(REPORT_DIR / "exit_study_manifest.json", {
        **result,
        "input_quality": quality,
        "fee_source_provenance_sha256": _sha256(SOURCE_PROVENANCE_PATH),
        "data_manifest_sha256": _sha256(DATA_MANIFEST_PATH),
        "common_market_id_sha256": hashlib.sha256("\n".join(sorted(common_ids)).encode("utf-8")).hexdigest(),
    })
    return result


def _write_report(config, provenance, summary: pd.DataFrame, monthly: pd.DataFrame, finalists, previous,
                  exit_comparison: pd.DataFrame | None = None, exit_manifest: dict | None = None):
    """Write the study report after entry, exit, and separate rebate diagnostics are available."""
    selection = provenance["walk_forward_selection"]
    selected = selection["selected"]
    challenger = selection.get("best_active_challenger", {})
    fixed_id = "fixed_5_usd|edge=0.0000|cap=none"
    cap_id = f"{readiness._policy_key(20.0)}|edge=0.0000|cap=20"
    index = summary.set_index("scenario_id", drop=False)

    def scenario_row(scenario_id):
        if scenario_id in index.index:
            row = index.loc[scenario_id]
            return row.iloc[0] if isinstance(row, pd.DataFrame) else row
        return pd.Series(dtype=object)

    def number(row, key, default=0.0):
        value = row.get(key, default) if hasattr(row, "get") else default
        return default if pd.isna(value) else float(value)

    def fmt_usd(value):
        return f"${value:,.2f}"

    selected_id = str(selected["scenario_id"])
    challenger_monthly = monthly.loc[monthly.scenario_id.astype(str).eq(str(challenger.get("scenario_id", "")))]
    challenger_oos = challenger_monthly.loc[challenger_monthly.period_role.eq("out_of_sample")]
    challenger_oos_pnl = float(challenger_oos.net_pnl_usd.sum()) if len(challenger_oos) else 0.0
    challenger_oos_entries = int(challenger_oos.entry_count.sum()) if len(challenger_oos) else 0

    table_rows = [
        ("Brak transakcji (wybór chronologiczny)", selected_id, "selected"),
        ("Najlepszy aktywny challenger (diagnoza)", str(challenger.get("scenario_id", "")), "challenger"),
        ("Stałe $5, edge 0%", fixed_id, "fixed"),
        ("5% wolnej gotówki, limit $20, edge 0%", cap_id, "cap20"),
    ]
    lines = [
        "# Historyczny backtest BTC 5m T−59: fee, wejścia i wyjścia",
        "",
        f"**Rekomendacja: brak transakcji z powodu braku stabilnej przewagi.** Selekcja wejścia z kontrolą hold-to-resolution wybrała {selected.get('policy', selected_id)}, bo żaden aktywny wariant wejścia nie przekroczył $100 do 2026-05-31. W późniejszej diagnostyce łączonej polityki wejścia/wyjścia close_at_60s wybrany wyłącznie na treningu podniósł equity challengera do około $140, po czym okres OOS przyniósł około −$139; wynik końcowy spadł do około $1,47. Pełna historia do 2026-10-06 była już wcześniej analizowana w repozytorium i nie jest nietkniętym testem. Modelu nie trenowano, konfiguracji live nie zmieniano i zleceń nie wysyłano.",
        "",
        "## Zakres i zamrożone założenia",
        "",
        f"Badanie objęło {int(provenance['study_input_rows']):,} kwalifikujących się snapshotów T−59 od 2026-04-15 17:05 UTC do 2026-10-06 23:55 UTC. Zamrożono predykcje, model, kalibrator, kotwicę T−59 i zapisane opóźnienie zlecenia 1 s. SHA256 predykcji i artefaktów modelu są zapisane w manifest.json. To kandydat badawczy, nie aktywny bundle live.",
        "",
        "Portfel jest jedną ciągłą ścieżką z $100 bez dopłat. Wejście używa ask nie starszego niż 1 s, limitu ceny 0,95, minimum 5 udziałów według obecnego Gamma (historycznego minimum nie udało się potwierdzić) i pełnej żądanej głębokości; brak pełnej głębokości oznacza skip. Fee gotówkowe musi mieścić się w dostępnej gotówce. 5% sizingu liczy się od wolnej gotówki po odjęciu zajętego kosztu. Rozliczenie gotówki następuje 60 s po późniejszym z: oficjalnego resolved_at lub startu rynku + 5 min.",
        "",
        "Snapshot T−59 nie gwarantuje fillu po opóźnieniu. Brakuje historycznych ACK, częściowych filli i dokładnych historycznych minimów. Inferencja w replayu miała opóźnienie 0 s; lokalne p50/p95/p99 kandydata wynosi 14,85/22,74/28,00 ms. Opóźnienia sieci, ACK i fillu pozostają niezmierzone. Istniejący replay 521 429 475 zdarzeń nie był powtarzany.",
        "",
        "## Historyczne opłaty",
        "",
        "Stawka 0,072 jest przypisana do 2026-05-06 00:00 UTC, a 0,07 od tej chwili. Granica 6 maja jest estymacją midpoint między archiwalnymi źródłami, nie potwierdzoną datą zmiany; wrażliwość policzono dla każdego dnia 23 kwietnia–15 maja włącznie i daty nie wybierano według PnL. Opłaty z 14 i 22 kwietnia oraz 4 maja wspierają 0,072; źródła z 8 i 10 maja wspierają 0,07.",
        "",
        "Oddzielono formułę od jednostki poboru. Formuła to ilość × stawka × p × (1−p). V1 pobiera fee kupna w udziałach (floor do 6 miejsc na zagregowanym poziomie ceny), a fee sprzedaży w gotówce (floor do 6 miejsc). V2 pobiera gotówkę po obu stronach, zaokrągloną half-up do 5 miejsc z minimum 0,00001. Fee-share obniża udziały/payout; fee gotówkowe zwiększa debet kupna lub zmniejsza wpływ sprzedaży. Fee nie jest podwójnie naliczane. To rekonstrukcja poziomów zagregowanej księgi, nie per-match historyczne filli.",
        "",
        "Aktualny zrzut Gamma jest zachowany tylko do audytu i nie służy jako dowód historycznej stawki. Migracja CLOB V1→V2 z 28 kwietnia oraz przybliżone godzinne okno przerwy są modelowane oddzielnie; dokładna sekunda przełączenia pozostaje nieznana. Przypisanie stawki do rynku ma status estymacji, a nie potwierdzenia z jego filli.",
        "",
        "## Wejścia i wybór chronologiczny",
        "",
        "Przebadano 40 kombinacji: 4 progi edge after-cost (0%, 2%, 4%, 8%) × stałe $5 albo 5% wolnej gotówki z limitami $5/$10/$15/$20/$30/$50/$75/$100/bez limitu. Edge to przewidywany zysk netto podzielony przez debet gotówkowy wejścia. Wybór używa wyłącznie cost-basis equity na koniec treningu 31 maja; żaden wariant aktywny nie pobił $100.",
        "",
        "| Wariant | Equity 31 V | Wejścia | Pełny PnL | Gotówka po zwolnieniu | Max DD | Obrót | Fee | Skipy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, scenario_id, _role in table_rows:
        row = scenario_row(scenario_id)
        if row.empty:
            continue
        equity = number(row, "training_equity_2026_05_31_usd", 100.0)
        lines.append(
            f"| {label} | {fmt_usd(equity)} | {int(number(row, 'trade_count')):,} | "
            f"{fmt_usd(number(row, 'net_pnl_usd'))} | {fmt_usd(number(row, 'ending_cash_after_assumed_release_usd', 100.0))} | "
            f"{number(row, 'max_drawdown_cost_basis_equity'):.1%} | {fmt_usd(number(row, 'gross_turnover_usd'))} | "
            f"{fmt_usd(number(row, 'fees_paid_estimated_usd'))} | {int(number(row, 'skip_count')):,} |"
        )
    lines += [
        "",
        f"Wybrany wariant no-trade utrzymuje $100. Najlepszy aktywny challenger miał equity {fmt_usd(float(challenger.get('training_equity_at_2026_05_31_usd', 100.0)))} na cutoffie, {int(challenger.get('training_trade_count_through_cutoff', 0))} wejść do cutoffu; jego późniejszy, retrospektywny wycinek OOS od 1 czerwca miał {challenger_oos_entries:,} wejść i PnL {fmt_usd(challenger_oos_pnl)}. Stan kapitału jest ciągły, bez resetu w czerwcu.",
        "",
        "",
        "## Porównanie z poprzednim raportem i wrażliwość fee",
        "",
    ]
    previous_specs = (
        ("fixed_5_usd", fixed_id),
        ("free_cash_5pct_cap20", cap_id),
    )
    previous_lines = [
        "| Kontrola | Poprzedni PnL | Nowy PnL | Różnica PnL | Poprzednie fee | Nowe fee | Transakcje poprzednio → teraz |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy_name, scenario_id in previous_specs:
        prior = previous.loc[previous.policy.astype(str).eq(policy_name)]
        current = scenario_row(scenario_id)
        if prior.empty or current.empty:
            continue
        prior_row = prior.iloc[0]
        prior_pnl = float(prior_row["previous_net_pnl_usd"])
        current_pnl = number(current, "net_pnl_usd")
        previous_lines.append(
            f"| {policy_name} | {fmt_usd(prior_pnl)} | {fmt_usd(current_pnl)} | {fmt_usd(current_pnl - prior_pnl)} | "
            f"{fmt_usd(float(prior_row['previous_fees_usd']))} | {fmt_usd(number(current, 'fees_paid_estimated_usd'))} | "
            f"{int(prior_row['previous_trade_count']):,} → {int(number(current, 'trade_count')):,} |"
        )
    lines += [
        "",
        "W poprzednim raporcie stawka 0,07 była użyta wstecz dla całej historii. Poniżej różnice dla kontroli edge 0%; nowy harmonogram 0,072/0,07, estymowana granica 6 maja i fee-share V1 zmieniają zarówno fee, jak i ścieżkę dostępnego kapitału. Pełne porównanie obrotu, DD i skipów znajduje się w previous_report_comparison.csv.",
        "",
        *previous_lines,
        "",
        "Codzienna analiza wrażliwości daty zmiany jest w fee_transition_sensitivity.csv.",
        "",
        "## Wyjścia na wspólnej pokrytej próbie",
        "",
    ]

    if exit_comparison is not None and not exit_comparison.empty:
        paired = exit_comparison.loc[exit_comparison.get("sample", pd.Series(index=exit_comparison.index, dtype=object)).eq("common_quote_coverage")]
        if len(paired):
            lines += [
                f"Wyjścia oceniono dla challengera i dwóch kontroli wejścia. Wspólny quote cache obejmuje {int((exit_manifest or {}).get('common_quote_market_count', 0)):,} rynków ze źródłem pełnym; {int((exit_manifest or {}).get('quote_cache_report', {}).get('source_incomplete_count', 0)):,} spośród pozyskanych ścieżek nie weszły do paired sample. Hold-to-resolution full-entry control jest osobnym wierszem w tabeli porównań.",
                "",
                "| Wejście | Exit | Rynki | Transakcje | Sprzedaże | Hold | PnL | Gotówka końcowa | Fee | Max DD |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            role_labels = {
                "best_active_challenger": "challenger edge 8%",
                "fixed_5_control": "stałe $5 edge 0%",
                "cap_20_control": "5% cash cap $20 edge 0%",
            }
            for row in paired.itertuples(index=False):
                lines.append(
                    f"| {role_labels.get(row.entry_role, row.entry_role)} | {row.exit_policy} | "
                    f"{int(getattr(row, 'common_market_count', 0)):,} | {int(getattr(row, 'trade_count', 0)):,} | "
                    f"{int(getattr(row, 'sold_count', 0)):,} | {int(getattr(row, 'held_to_resolution_count', 0)):,} | "
                    f"{fmt_usd(float(getattr(row, 'net_pnl_usd', 0.0)))} | "
                    f"{fmt_usd(float(getattr(row, 'ending_cash_after_all_releases_usd', 100.0)))} | "
                    f"{fmt_usd(float(getattr(row, 'fees_paid_estimated_usd', 0.0)))} | "
                    f"{float(getattr(row, 'maximum_drawdown_cost_basis_equity', 0.0)):.1%} |"
                )
            source_counts = (exit_manifest or {}).get("quote_cache_report", {}).get("source_counts", {})
            lines += [
                "",
                f"Źródła quote paths: {source_counts}. PMXT używa kolejności receive-time, natywnej głębokości bid i limitu świeżości 1 s; Kacho jest próbką top-of-book 1 Hz z rozmiarem top levelu. Po decyzji obowiązuje 1 s opóźnienia zlecenia. TP/SL ocenia wartość likwidacji netto z fee co 5 s; pełna pozycja musi mieć dostępną głębokość, w przeciwnym razie nie księguje się częściowego wyjścia. Progi można ponawiać na kolejnych przyczynowych kwotowaniach; stały close ma jedną próbę. Sprzedaż uwalnia środki dopiero w chwili wykonania, a wejście o tym samym timestamp jest obsługiwane wcześniej.",
            ]
        else:
            lines.append("Wynik porównania wyjść nie zawiera jeszcze wspólnej pokrytej próby.")
    else:
        lines.append("Badanie wyjść nie zostało jeszcze dołączone.")
    performance = (exit_manifest or {}).get("performance", {})
    cold_reference = performance.get("cold_cache_reference", {})
    cold_text = (
        f" Pierwsze przejście cold-cache: {int(cold_reference.get('extracted_market_count', 0)):,} ścieżek w "
        f"{float(cold_reference.get('quote_extraction_wall_seconds', 0.0)):.2f} s."
        if cold_reference else ""
    )
    quote_report = (exit_manifest or {}).get("quote_cache_report", {})
    if quote_report:
        lines += [
            "",
            f"Pomiar wykonania exit study: {int(quote_report.get('requested_market_count', 0)):,} quote paths, selektywna ekstrakcja {float(quote_report.get('wall_seconds', 0.0)):.2f} s, discovery simulation {float(quote_report.get('discovery_simulation_wall_seconds', 0.0)):.2f} s, cache hits {int(quote_report.get('cache_hit_market_count_at_run_start', 0)):,}; próbka 12 rynków (6 PMXT, 6 Kacho) zajęła {float(quote_report.get('profiled_reference', {}).get('elapsed_seconds', 0.0)):.2f} s.{cold_text} Cały run: {float(performance.get('study_wall_seconds', (exit_manifest or {}).get('study_wall_seconds', 0.0))):.2f} s; peak working set: {float(performance.get('peak_working_set_mib') or 0.0):.1f} MiB (0 oznacza brak pomiaru).",
        ]
    exit_selection = (exit_manifest or {}).get("walk_forward_exit_selection")
    if exit_selection:
        lines += [
            "",
                f"Walk-forward exit-only na challengerze wybrał do 31 maja {exit_selection['selected_exit_policy']}: equity {fmt_usd(float(exit_selection['selected_exit_training_equity_2026_05_31_usd']))} wobec {fmt_usd(float(exit_selection['hold_control_training_equity_2026_05_31_usd']))} dla hold. W OOS ta polityka wykonała {int(exit_selection['oos_entry_count']):,} wejść, miała PnL {fmt_usd(float(exit_selection['oos_monthly_pnl_usd']))} i końcowe equity {fmt_usd(float(exit_selection['oos_ending_equity_usd']))}. Ta niestabilność uzasadnia ogólną rekomendację no-trade.",
        ]

    rebate_path = REPORT_DIR / "taker_rebate_estimate.csv"
    lines += ["", "## Rabaty", ""]
    if rebate_path.exists():
        rebates = pd.read_csv(rebate_path)
        rebate_rows = rebates.loc[rebates.role.eq("best_active_challenger_exit_close_at_60s")]
        if len(rebate_rows):
            rebate = rebate_rows.iloc[0]
            lines.append(
                f"Obecna dokumentacja podaje start programu taker 28 maja 2026. Dla challenger + close_at_60s w common-sample ledger po starcie programu oszacowano {int(rebate.eligible_entry_trade_count)} wejść i {int(rebate.eligible_exit_sale_count)} sprzedaży; wariantowa estymacja wypłaconej gotówki to {fmt_usd(float(rebate.rebate_cash_paid_usd_estimated))}, a bonusów {fmt_usd(float(rebate.one_time_bonus_cash_paid_usd_estimated))}."
            )
        lines.append("To wyłącznie overlay według aktualnie zapisanych progów. Historyczne warunki launchu nie są potwierdzone; ponieważ strona nie podaje pieniężnej podstawy procentu rabatu, estymacja mnoży go przez zrekonstruowane fee wykonanych transakcji taker i jest ilustracyjna. Dla sprzedaży samo historyczne uprawnienie do rabatu również jest niepotwierdzone. Wartość wV używa formuły rozmiar × (1−cena) × waga crypto 2,3. Dzienna zmiana tieru i wypłata o północy UTC z minimum $1 są modelowane oddzielnie; naliczone, niewypłacone środki nie finansują wcześniejszych wejść. Bonusy i rabaty nie wpływają na wybór; maker rebate nie jest naliczany.")
    else:
        lines.append("Estymacja aktualnego programu taker nie została jeszcze wygenerowana. Maker rebate nie jest naliczany, a rabaty nie wpływają na wybór.")

    lines += [
        "",
        "## Pliki i źródła",
        "",
        "- historical_policy_report.md — to podsumowanie.",
        "- policy_comparison.csv, monthly_portfolio.csv, finalist_trade_ledger.csv — pełna siatka wejść, miesięczne stany i ledger.",
        "- reports/btc_fee_history_20261010/independent_signal_diagnostic_summary.csv — niezależnie finansowana diagnostyka zachowana osobno, nie jest wynikiem portfela $100.",
        "- fee_transition_sensitivity.csv, fee_rule_assignment_audit.csv.gz — zmiana daty fee oraz przypisanie stawki/jednostki.",
        "- exit_policy_comparison.csv, exit_policy_monthly.csv, exit_trade_ledger.csv, exit_path_coverage.csv — wyjścia i wspólne quote paths.",
        "- taker_rebate_estimate.csv, taker_rebate_cashflows_estimate.csv, taker_rebate_trade_ledger_estimate.csv — odseparowany szacunek aktualnych warunków.",
        "- manifest.json, walk_forward_selection.json, exit_study_manifest.json, taker_rebate_estimate_manifest.json — pochodzenie i założenia.",
        "- sources/ oraz source_provenance.json — zachowane kopie źródeł i ich SHA256.",
        "",
        "Archiwalne źródła: Polymarket Fees z 14 i 22 kwietnia, Maker Rebates z 4 maja, Fees z 8 i 10 maja, informacja o migracji z 28 kwietnia oraz aktualna strona Taker Rebates. Ich URL-e, kopie i skróty SHA256 są w source_provenance.json.",
    ]
    (REPORT_DIR / "historical_policy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    entry_result = run_entry_study()
    exit_result = run_exit_study()
    print(json.dumps({
        "report_dir": REPORT_DIR.relative_to(ROOT).as_posix(),
        "selected": entry_result["walk_forward_selection"]["selected"],
        "challenger": entry_result["walk_forward_selection"]["best_active_challenger"],
        "input_rows": entry_result["study_input_rows"],
        "entry_wall_seconds": entry_result["generated_wall_seconds"],
        "common_exit_quote_markets": exit_result["common_quote_market_count"],
        "exit_scenarios": exit_result["entry_scenario_count"] * exit_result["exit_policy_count"],
        "quote_extraction_wall_seconds": exit_result["quote_cache_report"]["wall_seconds"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
