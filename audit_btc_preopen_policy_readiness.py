"""Freeze and audit inputs for the offline BTC pre-open policy study."""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports/btc_preopen"
SNAPSHOT_PATH = REPORT_DIR / "entry_snapshots.parquet"
ELIGIBILITY_PATH = REPORT_DIR / "quote_validation_market_changes.csv"
FRESH_BBO_PATH = REPORT_DIR / "fresh_bbo_entry_results.csv"
CORRECTED_TRADES_PATH = REPORT_DIR / "quote_validation_corrected_trades.parquet"
PRIMARY_COMPARISON_PATH = REPORT_DIR / "primary_economic_comparison.csv"
EXTERNAL_PREDICTIONS_PATH = REPORT_DIR / "candidate_external_predictions.parquet"
ARTIFACT_MANIFEST_PATH = REPORT_DIR / "artifact_manifest.json"
OPPORTUNITIES_PATH = REPORT_DIR / "policy_opportunities.parquet"
STUDY_PATH = REPORT_DIR / "policy_study.json"
TRIALS_PATH = REPORT_DIR / "policy_trials.csv"
POLICY_PATH = REPORT_DIR / "policy.json"

ENTRY_CASE = "prestart_c0_o1"
MODEL = "candidate_platt"
INITIAL_CASH_USD = 100.0
BASELINE_GROSS_USD = 5.0
ASK_AGE_CAP_SECONDS = 30
RELEASE_DELAY_SECONDS = 60
MIN_RETURN_THRESHOLDS = (0.0, 0.02, 0.05)
FIXED_STAKES_USD = (1.0, 2.5)
FREE_CASH_FRACTIONS = (0.01, 0.025, 0.05)
FRACTIONAL_KELLY_MULTIPLIERS = (0.25, 0.5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _flatten_fill(row: pd.Series, side: str) -> dict:
    fill = row[f"{side}_fill"]
    fill = fill if isinstance(fill, dict) else {}
    return {
        f"{side}_fill_5usd_depth_sufficient": fill.get("depth_sufficient"),
        f"{side}_fill_5usd_gross_usd": fill.get("gross_usd"),
        f"{side}_fill_5usd_gross_shares": fill.get("gross_shares"),
        f"{side}_fill_5usd_net_shares": fill.get("shares"),
        f"{side}_fill_5usd_fee_usd": fill.get("fee_usd"),
        f"{side}_fill_5usd_fee_shares": fill.get("fee_shares"),
        f"{side}_fill_5usd_cash_fee_usd": fill.get("fee_cash_usd"),
        f"{side}_fill_5usd_cash_debit_usd": fill.get("cash_debit_usd"),
        f"{side}_fill_5usd_vwap": fill.get("vwap"),
    }


def _trial_rows() -> list[dict]:
    rows = [
        {"trial_id": "no_trade", "family": "no_trade", "parameters": {"stake_usd": 0.0}},
        {
            "trial_id": "baseline_fixed_5_positive_ev",
            "family": "current_fixed_stake_baseline",
            "parameters": {"stake_usd": BASELINE_GROSS_USD, "min_expected_return": 0.0},
        },
    ]
    for stake in FIXED_STAKES_USD:
        for minimum_return in MIN_RETURN_THRESHOLDS:
            rows.append({
                "trial_id": f"fixed_{stake:g}_roi_{minimum_return:g}",
                "family": "fixed_stake_expected_net_profit",
                "parameters": {"stake_usd": stake, "min_expected_return": minimum_return},
            })
    for minimum_return in MIN_RETURN_THRESHOLDS[1:]:
        rows.append({
            "trial_id": f"fixed_5_roi_{minimum_return:g}",
            "family": "fixed_stake_expected_net_profit",
            "parameters": {"stake_usd": BASELINE_GROSS_USD, "min_expected_return": minimum_return},
        })
    for fraction in FREE_CASH_FRACTIONS:
        for minimum_return in MIN_RETURN_THRESHOLDS:
            rows.append({
                "trial_id": f"free_cash_{fraction:g}_roi_{minimum_return:g}",
                "family": "free_cash_fraction",
                "parameters": {
                    "fraction_of_free_cash": fraction,
                    "max_gross_stake_usd": BASELINE_GROSS_USD,
                    "min_expected_return": minimum_return,
                },
            })
    for multiplier in FRACTIONAL_KELLY_MULTIPLIERS:
        for minimum_return in MIN_RETURN_THRESHOLDS:
            rows.append({
                "trial_id": f"fractional_kelly_{multiplier:g}_roi_{minimum_return:g}",
                "family": "fractional_kelly",
                "parameters": {
                    "kelly_multiplier": multiplier,
                    "max_gross_stake_usd": BASELINE_GROSS_USD,
                    "min_expected_return": minimum_return,
                    "stake_grid_usd": 0.01,
                },
            })
    return rows


def _fold_plan(eligible: pd.DataFrame) -> list[dict]:
    ordered = eligible.sort_values("entry_time_utc", kind="stable").reset_index(drop=True)
    count = len(ordered)
    edges = [int(count * value / 5) for value in range(6)]
    edges[-1] = count
    folds = []
    for fold_id, (validation_start, evaluation_start, evaluation_end) in enumerate(
        ((edges[1], edges[2], edges[3]), (edges[2], edges[3], edges[4]), (edges[3], edges[4], edges[5])),
        start=1,
    ):
        refit_at = ordered.iloc[evaluation_start].entry_time_utc
        validation = ordered.iloc[validation_start:evaluation_start]
        known_validation = validation[validation.outcome_available_at_utc < refit_at]
        past = ordered.iloc[:evaluation_start]
        known_past = past[past.outcome_available_at_utc < refit_at]
        evaluation = ordered.iloc[evaluation_start:evaluation_end]
        folds.append({
            "fold_id": fold_id,
            "refit_at_utc": refit_at.isoformat(),
            "selection_validation": {
                "start_index_in_eligible_order": validation_start,
                "end_index_exclusive": evaluation_start,
                "decision_start_utc": validation.entry_time_utc.min().isoformat(),
                "decision_end_utc": validation.entry_time_utc.max().isoformat(),
                "markets": len(validation),
                "labels_known_strictly_before_refit": len(known_validation),
            },
            "all_past_labels_known_strictly_before_refit": len(known_past),
            "evaluation": {
                "start_index_in_eligible_order": evaluation_start,
                "end_index_exclusive": evaluation_end,
                "decision_start_utc": evaluation.entry_time_utc.min().isoformat(),
                "decision_end_utc": evaluation.entry_time_utc.max().isoformat(),
                "markets": len(evaluation),
            },
            "capital_and_open_positions": "Carry continuously across outer blocks; no per-fold reset. A later settlement can influence a refit only after outcome_available_at_utc.",
        })
    return folds


def _baseline_ledger(
    trades: pd.DataFrame,
    expected_pnl: float,
    decision_start: pd.Timestamp,
    decision_end: pd.Timestamp,
) -> dict:
    entries = trades.copy()
    entries["entry_time_utc"] = pd.to_datetime(entries.entry_time_utc, utc=True)
    entries["capital_available_again_at_utc"] = pd.to_datetime(
        entries.capital_available_again_at_utc, utc=True
    )
    events = []
    for row in entries.itertuples(index=False):
        gross = float(row.gross_turnover_usd)
        events.append((row.entry_time_utc, 1, "entry", gross, float(row.cash_debit_usd), 0.0))
        events.append((row.capital_available_again_at_utc, 0, "release", gross, 0.0, float(row.payout_usd)))
    events.sort(key=lambda value: (value[0], value[1]))

    cash = INITIAL_CASH_USD
    locked = 0.0
    peak = INITIAL_CASH_USD
    peak_at = decision_start
    underwater_at = None
    max_underwater = pd.Timedelta(0)
    min_cash = cash
    max_exposure = 0.0
    max_positions = 0
    open_positions = 0
    max_stake_to_equity = 0.0
    max_drawdown = 0.0
    for timestamp, _, event, gross, debit, payout in events:
        if event == "entry":
            if cash + 1e-8 < debit:
                raise AssertionError("Baseline trade ledger spends unavailable cash")
            equity_before = cash + locked
            max_stake_to_equity = max(max_stake_to_equity, gross / equity_before)
            cash -= debit
            locked += gross
            open_positions += 1
        else:
            locked -= gross
            cash += payout
            open_positions -= 1
        if locked < -1e-7 or open_positions < 0:
            raise AssertionError("Baseline releases locked capital more than once")
        locked = max(0.0, locked)
        min_cash = min(min_cash, cash)
        max_exposure = max(max_exposure, locked)
        max_positions = max(max_positions, open_positions)
        equity = cash + locked
        if equity > peak + 1e-9:
            if underwater_at is not None:
                max_underwater = max(max_underwater, timestamp - underwater_at)
                underwater_at = None
            peak = equity
            peak_at = timestamp
        elif equity < peak - 1e-9:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
            if underwater_at is None:
                underwater_at = peak_at
    final_time = max(events[-1][0] if events else decision_end, decision_end)
    if underwater_at is not None:
        max_underwater = max(max_underwater, final_time - underwater_at)
    final_cash = cash
    pnl = final_cash - INITIAL_CASH_USD
    if not math.isclose(pnl, expected_pnl, rel_tol=0.0, abs_tol=1e-7):
        raise AssertionError(f"Baseline cash/PnL mismatch: {pnl} != {expected_pnl}")
    if open_positions != 0 or not math.isclose(locked, 0.0, abs_tol=1e-8):
        raise AssertionError("Baseline ledger did not release every position")
    daily_log_growth = []
    if events:
        event_index = 0
        daily_cash = INITIAL_CASH_USD
        daily_locked = 0.0
        day_start = decision_start.floor("D")
        day_end = final_time.floor("D")
        for day in pd.date_range(day_start, day_end, freq="D", tz="UTC"):
            cutoff = min(day + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1), final_time)
            while event_index < len(events) and events[event_index][0] <= cutoff:
                _, _, event, gross, debit, payout = events[event_index]
                if event == "entry":
                    daily_cash -= debit
                    daily_locked += gross
                else:
                    daily_locked -= gross
                    daily_cash += payout
                event_index += 1
            day_equity = daily_cash + daily_locked
            previous_equity = INITIAL_CASH_USD if not daily_log_growth else previous_day_equity
            if day_equity <= 0.0 or previous_equity <= 0.0:
                daily_log_growth.append(float("-inf"))
            else:
                daily_log_growth.append(math.log(day_equity / previous_equity))
            previous_day_equity = day_equity
    return {
        "initial_cash_usd": INITIAL_CASH_USD,
        "ending_cash_usd": final_cash,
        "net_pnl_usd": pnl,
        "max_drawdown_at_cost": max_drawdown,
        "maximum_underwater_duration_seconds": max_underwater.total_seconds(),
        "average_daily_log_growth": float(np.mean(daily_log_growth)) if daily_log_growth else 0.0,
        "daily_log_growth_days": len(daily_log_growth),
        "trade_count": len(entries),
        "gross_turnover_usd": float(entries.gross_turnover_usd.sum()),
        "fees_paid_usd": float(entries.fees_usd.sum()),
        "expected_net_value_at_entry_usd": float(entries.chosen_side_expected_net_value_usd.sum()),
        "max_open_cost_basis_usd": max_exposure,
        "max_open_positions": max_positions,
        "minimum_free_cash_after_entries_usd": min_cash,
        "max_stake_to_cost_basis_equity": max_stake_to_equity,
        "insufficient_cash_trades": int((entries.cash_available_before_usd < entries.cash_debit_usd - 1e-8).sum()),
    }


def _replay_baseline(data: pd.DataFrame, archived_trades: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cash = INITIAL_CASH_USD
    locked_cost = 0.0
    pending = []
    replayed = []
    sequence = 0
    counters = {"markets_seen": 0, "ineligible": 0, "no_positive_ev": 0, "insufficient_cash": 0}

    def settle(until: pd.Timestamp) -> None:
        nonlocal cash, locked_cost
        while pending and pending[0][0] <= until.value:
            _, _, gross, payout = heapq.heappop(pending)
            locked_cost -= gross
            cash += payout
        if abs(locked_cost) < 1e-9:
            locked_cost = 0.0

    for row in data.sort_values("entry_time_utc", kind="stable").itertuples(index=False):
        counters["markets_seen"] += 1
        entry_at = row.entry_time_utc
        settle(entry_at)
        if not row.eligible:
            counters["ineligible"] += 1
            continue
        p_up = float(row.p_candidate_platt)
        up_fill, down_fill = row.up_fill, row.down_fill
        if not up_fill["depth_sufficient"] or not down_fill["depth_sufficient"]:
            raise AssertionError("A corrected eligible market lacks its exact $5 side fill")
        up_ev = p_up * float(up_fill["shares"]) - float(up_fill["cash_debit_usd"])
        down_ev = (1.0 - p_up) * float(down_fill["shares"]) - float(down_fill["cash_debit_usd"])
        if max(up_ev, down_ev) <= 0.0:
            counters["no_positive_ev"] += 1
            continue
        side = "up" if up_ev >= down_ev else "down"
        fill = up_fill if side == "up" else down_fill
        debit = float(fill["cash_debit_usd"])
        if cash + 1e-9 < debit:
            counters["insufficient_cash"] += 1
            continue
        outcome = int(row.target_polymarket_up)
        payout = float(fill["shares"]) if outcome == (1 if side == "up" else 0) else 0.0
        gross = float(fill["gross_usd"])
        ev = up_ev if side == "up" else down_ev
        before = cash
        cash -= debit
        locked_cost += gross
        release_at = row.capital_release_at_utc
        sequence += 1
        heapq.heappush(pending, (release_at.value, sequence, gross, payout))
        replayed.append({
            "condition_id": row.condition_id,
            "chosen_side": side,
            "entry_time_utc": entry_at,
            "capital_available_again_at_utc": release_at,
            "gross_turnover_usd": gross,
            "cash_debit_usd": debit,
            "cash_available_before_usd": before,
            "cash_available_after_usd": cash,
            "fees_usd": float(fill["fee_usd"]),
            "payout_usd": payout,
            "net_pnl_usd": payout - debit,
            "chosen_side_expected_net_value_usd": ev,
        })
    settle(pd.Timestamp.max.tz_localize("UTC"))
    replayed_frame = pd.DataFrame(replayed)
    archived = archived_trades.sort_values("entry_time_utc", kind="stable").reset_index(drop=True)
    if not replayed_frame.condition_id.equals(archived.condition_id.reset_index(drop=True)):
        raise AssertionError("Replayed baseline markets differ from the corrected executed-trade ledger")
    if not replayed_frame.chosen_side.equals(archived.chosen_side.reset_index(drop=True)):
        raise AssertionError("Replayed baseline side choice differs from the corrected executed-trade ledger")
    comparison_columns = (
        "entry_time_utc", "capital_available_again_at_utc", "gross_turnover_usd",
        "cash_debit_usd", "cash_available_before_usd", "cash_available_after_usd",
        "fees_usd", "payout_usd", "net_pnl_usd", "chosen_side_expected_net_value_usd",
    )
    archived_names = {"cash_available_after_usd": "cash_available_after_entry_usd"}
    for column in comparison_columns:
        archived_column = archived_names.get(column, column)
        if column.endswith("_utc"):
            actual = pd.to_datetime(replayed_frame[column], utc=True).astype("int64").to_numpy()
            expected = pd.to_datetime(archived[archived_column], utc=True).astype("int64").to_numpy()
            matches = np.array_equal(actual, expected)
        else:
            matches = np.allclose(
                replayed_frame[column].to_numpy(dtype=float),
                archived[archived_column].to_numpy(dtype=float), rtol=0.0, atol=1e-8,
            )
        if not matches:
            raise AssertionError(f"Replayed baseline differs from archived ledger field: {column}")
    return replayed_frame, counters


def run() -> None:
    snapshots = pd.read_parquet(SNAPSHOT_PATH)
    snapshots = snapshots[snapshots.entry_case.eq(ENTRY_CASE)].copy()
    if snapshots.condition_id.duplicated().any():
        raise AssertionError("Primary entry cache has duplicate markets")
    snapshots["entry_time_utc"] = pd.to_datetime(snapshots.entry_time_utc, utc=True)
    snapshots["market_start_utc"] = pd.to_datetime(snapshots.market_start_utc, utc=True)
    snapshots["resolved_at_utc"] = pd.to_datetime(snapshots.resolved_at_utc, utc=True)

    eligibility = pd.read_csv(ELIGIBILITY_PATH)
    if eligibility.condition_id.duplicated().any():
        raise AssertionError("Corrected eligibility cache has duplicate markets")
    fresh_bbo = pd.read_csv(FRESH_BBO_PATH)
    fresh_bbo = fresh_bbo[fresh_bbo.entry_case.eq(ENTRY_CASE)]
    if fresh_bbo.condition_id.duplicated().any():
        raise AssertionError("Corrected BBO cache has duplicate primary decisions")
    fresh_bbo = fresh_bbo.rename(columns={
        "bbo_at_entry_ask_checks": "bbo_at_entry_ask_checks_corrected",
        "bbo_at_entry_ask_mismatches": "bbo_at_entry_ask_mismatches_corrected",
        "bbo_at_entry_not_newer_than_side_state": "bbo_at_entry_not_newer_than_side_state_corrected",
        "bbo_at_entry_older_than_side_state": "bbo_at_entry_older_than_side_state_corrected",
        "bbo_at_entry_tied_with_side_state": "bbo_at_entry_tied_with_side_state_corrected",
    })
    candidate_predictions = pd.read_parquet(
        EXTERNAL_PREDICTIONS_PATH, columns=["condition_id", "p_candidate_platt"]
    )
    prediction_rows = len(candidate_predictions)
    prediction_rows_without_market_id = int(candidate_predictions.condition_id.isna().sum())
    candidate_predictions = candidate_predictions.dropna(subset=["condition_id"]).copy()
    if candidate_predictions.condition_id.duplicated().any():
        raise AssertionError("Frozen prediction cache has duplicate markets")

    data = snapshots.merge(
        eligibility[[
            "condition_id", "reason_after_fresh_bbo",
            "eligible_locked_quotes_corrected_bbo_freshness",
            "ask_mismatches_after_fresh_bbo", "quote_valid_after_fresh_bbo",
        ]], on="condition_id", how="left", validate="one_to_one",
    ).merge(
        fresh_bbo[[
            "condition_id", "bbo_at_entry_ask_checks_corrected", "bbo_at_entry_ask_mismatches_corrected",
            "bbo_at_entry_not_newer_than_side_state_corrected", "bbo_at_entry_older_than_side_state_corrected",
            "bbo_at_entry_tied_with_side_state_corrected",
        ]], on="condition_id", how="left", validate="one_to_one",
    ).merge(
        candidate_predictions, on="condition_id", how="left", validate="one_to_one",
        suffixes=("", "_external"),
    )
    if len(data) != 9407 or data.reason_after_fresh_bbo.isna().any():
        raise AssertionError("Primary opportunities do not align to the full corrected market set")
    if data.p_candidate_platt.isna().any() or not np.allclose(
        data.p_candidate_platt, data.p_candidate_platt_external, rtol=0.0, atol=1e-14
    ):
        raise AssertionError("Primary predictions differ from the frozen external prediction cache")
    if data.target_polymarket_up.isna().any() or not data.target_polymarket_up.isin([0, 1]).all():
        raise AssertionError("Official outcomes are missing or invalid in the economic cache")

    market_end = data.market_start_utc + pd.Timedelta(minutes=5)
    data["outcome_available_at_utc"] = data.resolved_at_utc.where(
        data.resolved_at_utc >= market_end, market_end
    )
    data["capital_release_at_utc"] = data.outcome_available_at_utc + pd.Timedelta(
        seconds=RELEASE_DELAY_SECONDS
    )
    data["eligible"] = data.eligible_locked_quotes_corrected_bbo_freshness.astype(bool)
    data["bbo_confirmation_status"] = np.where(
        data.bbo_at_entry_ask_checks_corrected.fillna(0).gt(0),
        "checked_against_newer_reported_bbo",
        "no_decisive_bbo_confirmation",
    )
    for side in ("up", "down"):
        data[f"{side}_top_ask_capacity_usd"] = data[f"{side}_best_ask"] * data[
            f"{side}_best_ask_size_shares"
        ]

    opportunity_columns = [
        "condition_id", "market_slug", "market_start_utc", "entry_time_utc",
        "prediction_available_at_utc", "outcome_available_at_utc", "capital_release_at_utc",
        "p_candidate_platt", "target_polymarket_up", "resolved_at_utc", "eligible",
        "reason_after_fresh_bbo", "fee_collection_mode", "fee_known", "fee_rate_bps",
        "fee_age_seconds", "quote_valid_after_fresh_bbo", "ask_mismatches_after_fresh_bbo",
        "bbo_at_entry_ask_checks_corrected", "bbo_at_entry_ask_mismatches_corrected",
        "bbo_at_entry_not_newer_than_side_state_corrected", "bbo_at_entry_older_than_side_state_corrected",
        "bbo_at_entry_tied_with_side_state_corrected", "bbo_confirmation_status",
        "no_future_event_at_entry", "no_future_source_event_at_entry",
        "up_best_bid", "up_best_ask", "up_best_ask_size_shares", "up_top_ask_capacity_usd",
        "up_ask_age_seconds", "up_quote_source_token_id", "up_quote_complemented",
        "down_best_bid", "down_best_ask", "down_best_ask_size_shares", "down_top_ask_capacity_usd",
        "down_ask_age_seconds", "down_quote_source_token_id", "down_quote_complemented",
    ]
    opportunity_frame = data[opportunity_columns].copy()
    for side in ("up", "down"):
        flattened = pd.DataFrame([_flatten_fill(row, side) for _, row in data.iterrows()], index=data.index)
        opportunity_frame = opportunity_frame.join(flattened)
    opportunity_frame["ask_level_ladder_stored"] = False
    opportunity_frame["execution_depth_stored_as"] = "best ask/size plus exact aggregate fill at $5 gross"
    temporary_parquet = OPPORTUNITIES_PATH.with_suffix(".tmp.parquet")
    opportunity_frame.to_parquet(temporary_parquet, index=False, compression="zstd")
    temporary_parquet.replace(OPPORTUNITIES_PATH)

    corrected_trades = pd.read_parquet(CORRECTED_TRADES_PATH)
    baseline_trades = corrected_trades[
        corrected_trades.validation_stage.eq("locked_quotes_corrected_bbo_freshness")
        & corrected_trades.model.eq(MODEL)
    ].copy()
    primary = pd.read_csv(PRIMARY_COMPARISON_PATH)
    primary_row = primary[
        primary.sample_scope.eq("all_archived_markets")
        & primary.model.eq(MODEL)
        & primary.entry_case.eq(ENTRY_CASE)
        & primary.max_ask_age_seconds.eq(ASK_AGE_CAP_SECONDS)
        & primary.settlement_release_delay_seconds.eq(RELEASE_DELAY_SECONDS)
    ]
    if len(primary_row) != 1:
        raise AssertionError("The corrected T-59 baseline summary is missing or ambiguous")
    expected_pnl = float(primary_row.iloc[0].net_pnl_usd)
    replayed_trades, baseline_decisions = _replay_baseline(data, baseline_trades)
    baseline = _baseline_ledger(
        replayed_trades, expected_pnl, data.entry_time_utc.min(), data.entry_time_utc.max()
    )
    if len(baseline_trades) != 2316 or not math.isclose(expected_pnl, 475.84015563503385, abs_tol=1e-8):
        raise AssertionError("Corrected fixed-$5 baseline no longer matches the published reference")
    if baseline_trades.condition_id.duplicated().any():
        raise AssertionError("Baseline bought the same market more than once")
    if not baseline_trades.gross_turnover_usd.eq(BASELINE_GROSS_USD).all():
        raise AssertionError("Baseline contains a non-$5 gross order")
    if not math.isclose(
        baseline["max_drawdown_at_cost"], float(primary_row.iloc[0].max_drawdown_at_cost),
        rel_tol=0.0, abs_tol=1e-9,
    ):
        raise AssertionError("Reconstructed baseline drawdown differs from the published reference")

    trades_with_outcomes = baseline_trades.merge(
        data[["condition_id", "market_start_utc", "resolved_at_utc", "up_best_ask",
              "up_best_ask_size_shares", "down_best_ask", "down_best_ask_size_shares"]],
        on="condition_id", how="left", validate="one_to_one",
    )
    selected_capacity = trades_with_outcomes.apply(
        lambda row: float(row[f"{row.chosen_side}_best_ask"])
        * float(row[f"{row.chosen_side}_best_ask_size_shares"]), axis=1,
    )
    release_semantics_match = bool(
        (pd.to_datetime(baseline_trades.capital_available_again_at_utc, utc=True)
         == (data.set_index("condition_id").loc[baseline_trades.condition_id, "outcome_available_at_utc"].reset_index(drop=True)
             + pd.Timedelta(seconds=RELEASE_DELAY_SECONDS))).all()
    )
    if not release_semantics_match:
        raise AssertionError("Baseline release timestamps differ from max(resolution, market end) + release delay")

    eligible = data[data.eligible].sort_values("entry_time_utc", kind="stable")
    folds = _fold_plan(eligible)
    reason_counts = data.reason_after_fresh_bbo.value_counts().to_dict()
    eligible_count = int(data.eligible.sum())
    top_capacity = {
        f"both_sides_cover_{amount:g}_usd": int((
            data.loc[data.eligible, "up_top_ask_capacity_usd"].ge(amount)
            & data.loc[data.eligible, "down_top_ask_capacity_usd"].ge(amount)
        ).sum())
        for amount in (1.0, 2.0, 2.5, 5.0)
    }
    bbo_unconfirmed = int((data.loc[data.eligible, "bbo_confirmation_status"] == "no_decisive_bbo_confirmation").sum())
    trial_rows = _trial_rows()
    trial_frame = pd.DataFrame([
        {"trial_id": item["trial_id"], "family": item["family"],
         "parameters_json": json.dumps(item["parameters"], sort_keys=True),
         "status": "not_run_input_depth_gate", "validation_score": np.nan,
         "reason": "Per-entry ask-level ladders and historical minimum-order sizes are absent; variable stakes cannot be priced and validated without assumptions."}
        for item in trial_rows
    ])
    temporary_csv = TRIALS_PATH.with_suffix(".tmp.csv")
    trial_frame.to_csv(temporary_csv, index=False)
    temporary_csv.replace(TRIALS_PATH)

    artifact_manifest = json.loads(ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))
    candidate = artifact_manifest["candidate"]
    pmxt = artifact_manifest["pmxt_cache"]
    source_paths = [
        SNAPSHOT_PATH, ELIGIBILITY_PATH, FRESH_BBO_PATH, CORRECTED_TRADES_PATH,
        PRIMARY_COMPARISON_PATH, EXTERNAL_PREDICTIONS_PATH, ARTIFACT_MANIFEST_PATH,
        ROOT / "configs/btc_preopen_v1.json", ROOT / "configs/runtime/trade_policy_project.json",
        ROOT / "run_btc_preopen_economic_replay.py", ROOT / "utils/polymarket.py",
        ROOT / "audit_btc_preopen_policy_readiness.py",
    ]
    source_identity = {path.relative_to(ROOT).as_posix(): {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
                       for path in source_paths}
    model_path = ROOT / candidate["model_path"]
    calibrator_path = ROOT / candidate["calibrator_path"]
    model_identity = {
        "model_name": MODEL,
        "model_path": candidate["model_path"],
        "model_sha256": _sha256(model_path) if model_path.is_file() else candidate.get("model_hash_from_candidate_study"),
        "manifest_model_hash": candidate.get("model_hash_from_candidate_study"),
        "calibrator_path": candidate["calibrator_path"],
        "calibrator_sha256": _sha256(calibrator_path) if calibrator_path.is_file() else None,
        "ordered_feature_sha256_newline_utf8": candidate.get("ordered_feature_sha256_newline_utf8"),
        "prediction_path": EXTERNAL_PREDICTIONS_PATH.relative_to(ROOT).as_posix(),
        "prediction_column": "p_candidate_platt",
    }
    study = {
        "experiment_id": "btc_preopen_policy_t59_candidate_platt_20261007",
        "status": "not_run_input_depth_gate",
        "evaluation_label": "retrospective_walk_forward_policy_assessment; the period has prior development exposure",
        "entry": {"scenario": ENTRY_CASE, "relative_to_market_start": "T-59s", "ask_age_cap_seconds": ASK_AGE_CAP_SECONDS},
        "coverage": {
            "archived_markets": len(data), "corrected_eligible_markets": eligible_count,
            "qualification_reasons": reason_counts,
            "first_market_start_utc": data.market_start_utc.min().isoformat(),
            "last_market_start_utc": data.market_start_utc.max().isoformat(),
            "first_decision_utc": data.entry_time_utc.min().isoformat(),
            "last_decision_utc": data.entry_time_utc.max().isoformat(),
            "last_outcome_available_utc": data.outcome_available_at_utc.max().isoformat(),
            "eligible_books_without_decisive_bbo_confirmation": bbo_unconfirmed,
            "frozen_prediction_rows": prediction_rows,
            "prediction_rows_without_market_id": prediction_rows_without_market_id,
        },
        "depth_audit": {
            "stored_native_ask_levels": ["best ask price", "best ask size"],
            "full_ask_level_ladders_stored": False,
            "additional_stored_execution_point": "Exact aggregate fee-inclusive fill for $5 gross per side only",
            "top_level_capacity_on_eligible_markets": top_capacity,
            "baseline_selected_trades_requiring_levels_beyond_best_ask": int((selected_capacity < BASELINE_GROSS_USD - 1e-8).sum()),
            "baseline_selected_trade_count": len(baseline_trades),
            "reason_tuning_stopped": "Variable stakes below $5 can cross unrecorded intermediate ask levels; stakes above $5 exceed recorded execution depth. Historical minimum-order sizes are also absent. Reconstructing levels requires replaying PMXT events, which this task forbids.",
        },
        "frozen_search_space": {
            "trial_budget": len(trial_rows), "trials_run": 0, "search_method": "finite deterministic grid; no Optuna",
            "seed": None, "seed_reason": "No stochastic search or trials were run",
            "families": {
                "current_baseline": {"gross_stake_usd": BASELINE_GROSS_USD, "minimum_expected_return": 0.0},
                "no_trade": {"stake_usd": 0.0, "daily_log_growth": 0.0},
                "fixed_stake_expected_net_profit": {"gross_stakes_usd": list(FIXED_STAKES_USD), "min_return_grid": list(MIN_RETURN_THRESHOLDS)},
                "fixed_5_usd_min_return": {"gross_stake_usd": BASELINE_GROSS_USD, "min_return_grid": list(MIN_RETURN_THRESHOLDS[1:])},
                "free_cash_fraction": {"basis": "free cash only", "fractions": list(FREE_CASH_FRACTIONS), "max_gross_stake_usd": BASELINE_GROSS_USD, "min_return_grid": list(MIN_RETURN_THRESHOLDS)},
            "fractional_kelly": {"multipliers": list(FRACTIONAL_KELLY_MULTIPLIERS), "max_gross_stake_usd": BASELINE_GROSS_USD, "stake_grid_usd": 0.01, "min_return_grid": list(MIN_RETURN_THRESHOLDS)},
            },
            "minimum_expected_return_definition": "Expected net USD payout after historical fees divided by full cash debit, both evaluated at actual gross stake.",
            "range_rationale": {
                "fixed_stakes": "$1 and $2.50 plus the exact $5 baseline span 1%, 2.5%, and 5% of the $100 initial cash; amounts above $5 exceed the only stored aggregate execution point.",
                "free_cash_fractions": "1%, 2.5%, and 5% of free cash target the same initial-dollar band while allowing bankroll-dependent sizing; use free cash, never locked cost or possible payout, and cap gross stake at $5.",
                "kelly": "Quarter- and half-Kelly are bounded fractional-growth candidates; each uses actual fee/depth costs, free-cash affordability, a $5 cap, and one-cent stake increments.",
                "minimum_return": "0%, 2%, and 5% of full debit span merely positive net EV through modest after-cost hurdles without adding dozens of independent thresholds.",
            },
            "side_rule": "For fixed and free-cash sizing, choose the side with the greater positive expected net USD payout at that actual stake; skip if neither meets the threshold. Kelly separately maximizes expected log growth with actual fee/depth constraints.",
            "kelly_objective": "For stake s, maximize p*ln((free_cash - cash_debit(s) + net_shares(s))/free_cash) + (1-p)*ln((free_cash - cash_debit(s))/free_cash), then apply the stated fractional multiplier. Never count locked positions or possible payouts as free cash.",
            "stake_rounding": "Nearest cent, matching the existing trade-intent sizing precision.",
            "depth_limit": "No stake above $5. A non-$5 stake is evaluable only where its full ask ladder is present; top-level-only records cannot stand in for deeper asks.",
        },
        "selection_objective": "Mean daily log growth over inner chronological validation blocks, with no-trade growth zero; report PnL/final capital, cost-basis drawdown and duration underwater, trades, turnover, fees, exposure, stake-to-capital, depth/cash skips, and inability to place further orders.",
        "split_design_frozen_before_search": {
            "method": "Sort corrected eligible markets by entry_time_utc. First 20% is initial history; each outer block is 20%, preceded by a separate 20% policy-selection validation block.",
            "initial_history_count": int(len(eligible) * 0.2),
            "outer_folds": folds,
            "availability_rule": "Only labels with max(resolved_at_utc, market_start_utc + 5 minutes) strictly before the refit time enter selection.",
            "crossing_positions": "Carry cost basis and unsettled positions through block boundaries; cash is released only at max(resolved_at_utc, expiry) + 60 seconds.",
        },
        "baseline_reproduction": baseline,
        "baseline_decision_counts": baseline_decisions,
        "baseline_release_matches_resolution_and_expiry_rule": release_semantics_match,
        "trial_runtime_profile": {
            "trial_seconds": None,
            "projected_total_seconds": None,
            "reason": "No policy trial was run because exact variable-stake executions and historical minimum-order feasibility cannot be reconstructed from the available cache.",
        },
        "execution_environment": {
            "platform": "Windows",
            "logical_cpu_count": 20,
            "physical_memory_total_bytes": 68366831616,
            "active_background_processes": [
                "run_btc_preopen_collection.py",
                "run_btc_entry_state_collection.py",
            ],
            "policy_workers": 0,
            "gpu_used": False,
        },
        "baseline_reported_pnl_usd": expected_pnl,
        "policy_chosen": None,
        "reason_policy_not_selected": "The execution-depth cache lacks intervening ask levels and historical minimum-order sizes, so sizing trials would rely on unobserved prices or unsupported order-size assumptions.",
        "inputs": {
            "candidate": model_identity,
            "shared_market_evaluation_sha256": pmxt["archive_identity"]["source_sha256"],
            "pmxt_partition_list_sha256": pmxt["partition_list_hash_sha256"],
            "pmxt_event_rows": pmxt["event_rows"],
            "files": source_identity,
        },
    }
    _write_json(STUDY_PATH, study)

    policy = {
        "schema_version": 1,
        "policy_id": "candidate_platt_t59_fixed_5_positive_expected_net_payout_reference",
        "status": "retained_reference_only; sizing optimization gated by incomplete depth cache",
        "active_for_trading": False,
        "selection_result": "No sizing candidate was selected; do not describe the fixed-$5 reference as an optimized policy.",
        "model": model_identity,
        "decision": {
            "decision_time": "single decision per market using frozen candidate_platt probability and T-59 book snapshot",
            "probability": "p_candidate_platt is unchanged; no BTC model fit or recalibration in this stage",
            "eligible": "Corrected BBO/freshness qualification, both-side $5 depth valid, known historical fee, ask age <= 30 seconds, and no future event at entry",
            "extra_buffer_usd_per_share": 0.0,
            "gross_stake_usd": BASELINE_GROSS_USD,
            "execution_price": "Walk the ask-side full depth for a $5 gross order; no midpoint, future quote, or additional slippage assumption.",
            "expected_net_payout_usd": "p_side * net_shares_after_historical_fee - full_cash_debit_usd",
            "expected_value_metrics": {
                "ev_per_net_share_usd": "p_side - full_cash_debit_usd / net_shares_after_historical_fee",
                "ev_usd_per_order": "p_side * net_shares_after_historical_fee - full_cash_debit_usd",
                "expected_return": "ev_usd_per_order / full_cash_debit_usd",
            },
            "side_selection": "Compute both sides from the exact aggregate $5 ask-depth fill; buy the side with the larger positive expected net payout. If neither is positive, no_trade. Ties choose UP.",
            "expected_value_units": "USD per order; not USD per share",
        },
        "fees": {
            "historical_mode": "outcome-share fees before 2026-04-28 11:00 UTC; maintenance exclusion 11:00–12:00 UTC; collateral fee from 12:00 UTC",
            "legacy": "At each aggregate ask level, fee_shares = shares * rate * min(price, 1-price) / price, floored to six share decimals; payout uses gross shares minus fee shares.",
            "collateral": "fee_cash = shares * rate * (price * (1-price))^1; round to five decimals, apply the historical minimum fee, and include it in cash debit.",
            "maker_fill_precision": "The archive does not expose individual maker matches, so maker-level fee rounding cannot be reproduced.",
        },
        "portfolio": {
            "initial_free_cash_usd": INITIAL_CASH_USD,
            "one_purchase_per_market": True,
            "maximum_purchases_per_market": 1,
            "hold_to_resolution": True,
            "cash_constraint": "Require free cash for the full cash debit; no borrowing, deposits, or credit for unresolved payouts.",
            "capital_release": "max(resolved_at_utc, market_start_utc + 5 minutes) + 60 seconds",
            "equity_and_drawdown": "Free cash plus gross cost basis of locked positions; no mark-to-market while unresolved.",
            "historical_minimum_order": "Unavailable; no current minimum was projected backward.",
        },
        "evaluation": {"gross_pnl_usd": baseline["net_pnl_usd"], "trade_count": baseline["trade_count"], "walk_forward_candidate": None},
        "inputs": study["inputs"],
    }
    _write_json(POLICY_PATH, policy)
    print(json.dumps({
        "status": study["status"], "markets": len(data), "eligible": eligible_count,
        "qualification_reasons": reason_counts, "depth_audit": study["depth_audit"],
        "baseline": baseline, "folds": folds,
        "artifacts": [str(OPPORTUNITIES_PATH.relative_to(ROOT)), str(STUDY_PATH.relative_to(ROOT)),
                      str(TRIALS_PATH.relative_to(ROOT)), str(POLICY_PATH.relative_to(ROOT))],
    }, indent=2))


if __name__ == "__main__":
    run()
