"""Replay the frozen BTC confirmation subset under explicit resolution accounting."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

import run_btc_preopen_economic_replay as economic_replay
import run_btc_preopen_t59_reassessment as t59_reassessment
from utils.btc_resolution_portfolio import ResolutionPortfolio


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "configs/research/btc_resolution_development_20261026.json"
LEDGER_PATH = ROOT / "data/analysis/polymarket/BTC/btc_candidate_polymarket_target_20261023/economic_decision_ledger.parquet"
T60_PATH = ROOT / "reports/btc_preopen/entry_snapshots.parquet"
T59_PATH = ROOT / "data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/t59_ask_ladders.parquet"
OUT_ROOT = ROOT / "data/analysis/polymarket/BTC"
CANDIDATES = ("A", "B", "C", "D")
INITIAL_CASH_USD = 100.0
FIXED_GROSS_USD = 5.0
FREE_CASH_FRACTION = 0.05
FALLBACK_MINIMUM_USD = 5.0
FILL_TOLERANCE_USD = 1e-6
DEVELOPMENT_CUTOFF_UTC = pd.Timestamp("2026-07-01T00:00:00Z")
DEVELOPMENT_END_UTC = DEVELOPMENT_CUTOFF_UTC
T60_ENTRY_CASE = "prestart_c0_o0"
T59_ENTRY_CASE = "t59_native_asks"
T60_COLUMNS = [
    "condition_id", "market_start_utc", "entry_case", "entry_time_utc",
    "fee_collection_mode", "fee_rate_bps", "fee_known", "has_full_snapshot",
    "quote_valid", "depth_5usd_valid_both_sides", "no_future_event_at_entry",
    "no_future_source_event_at_entry", "bbo_at_entry_ask_mismatches",
    "up_best_ask", "up_best_ask_size_shares", "down_best_ask",
    "down_best_ask_size_shares",
]
T59_COLUMNS = [
    "condition_id", "market_start_utc", "entry_case", "fee_collection_mode",
    "fee_rate_bps", "fee_known", "has_full_snapshot", "quote_valid",
    "archive_entry_hour_available", "archive_freshness_window_available",
    "archive_window_missing_source_hours", "archive_window_unprocessed_hours",
    "no_future_event_at_entry", "no_future_source_event_at_entry",
    "archive_gap_after_fee_update", "archive_gap_after_up_book_init",
    "archive_gap_after_down_book_init", "bbo_at_entry_ask_mismatches",
    "depth_5usd_valid_both_sides", "up_ask_levels", "down_ask_levels",
]
T59_REQUIRED_TRUE = (
    "fee_known", "has_full_snapshot", "quote_valid",
    "archive_entry_hour_available", "archive_freshness_window_available",
    "no_future_event_at_entry", "no_future_source_event_at_entry",
    "depth_5usd_valid_both_sides",
)
T59_REQUIRED_FALSE = (
    "archive_gap_after_fee_update", "archive_gap_after_up_book_init",
    "archive_gap_after_down_book_init",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_protocol_inputs(protocol: dict) -> dict[str, str]:
    checked = {}
    protocol_manifest = ROOT / "reports/btc_preopen/btc_candidate_polymarket_target_20261023_v3_manifest.json"
    expected_manifest_hash = protocol["source_protocol_manifest_sha256"]
    actual_manifest_hash = sha256(protocol_manifest)
    if actual_manifest_hash != expected_manifest_hash:
        raise RuntimeError("Frozen E0031 protocol manifest hash changed")
    checked[protocol_manifest.relative_to(ROOT).as_posix()] = actual_manifest_hash

    for item in protocol["inputs"].values():
        path = ROOT / item["path"]
        actual = sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Input hash mismatch: {item['path']}")
        checked[item["path"]] = actual

    for key, item in protocol["fees"].items():
        if key.endswith("_path"):
            path = ROOT / item
            expected = protocol["fees"][key.replace("_path", "_sha256")]
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(f"Fee input hash mismatch: {item}")
            checked[item] = actual

    for path_key, hash_key in (
        ("fee_execution_helper_path", "fee_execution_helper_sha256"),
        ("book_source_helper_path", "book_source_helper_sha256"),
        ("resolution_ledger_helper_path", "resolution_ledger_helper_sha256"),
        ("analysis_runner_path", "analysis_runner_sha256"),
    ):
        item = protocol["inputs"]["source_code"][path_key]
        actual = sha256(ROOT / item)
        if actual != protocol["inputs"]["source_code"][hash_key]:
            raise RuntimeError(f"Source helper hash mismatch: {item}")
        checked[item] = actual
    return checked


def _verify_confirmation_ledger(ledger: pd.DataFrame, protocol: dict) -> None:
    period = protocol["period"]
    if len(ledger) != int(period["input_rows"]):
        raise RuntimeError(f"Expected {period['input_rows']} saved rows, found {len(ledger)}")
    if set(ledger["period"].astype(str)) != {period["name"]}:
        raise RuntimeError("Economic ledger contains rows outside the frozen confirmation subset")
    starts = pd.to_datetime(ledger["market_start_utc"], utc=True)
    if (starts < pd.Timestamp(period["market_start_inclusive_utc"])).any():
        raise RuntimeError("Economic ledger contains a row before the development interval")
    if (starts >= DEVELOPMENT_CUTOFF_UTC).any():
        raise RuntimeError("Economic ledger contains a protected holdout row")
    if starts.min() != pd.Timestamp(period["input_market_start_min_utc"]):
        raise RuntimeError("Economic ledger minimum market time does not match the freeze")
    if starts.max() != pd.Timestamp(period["input_market_start_max_utc"]):
        raise RuntimeError("Economic ledger maximum market time does not match the freeze")
    if ledger["condition_id"].astype(str).str.lower().duplicated().any():
        raise RuntimeError("Economic decision ledger has duplicate condition IDs")


def _read_t60(ids: set[str], expected_rows: int) -> dict[str, dict]:
    dataset = ds.dataset(T60_PATH, format="parquet")
    cutoff = pa.scalar(
        DEVELOPMENT_CUTOFF_UTC.to_pydatetime(),
        type=dataset.schema.field("market_start_utc").type,
    )
    filter_expr = (
        ds.field("condition_id").isin(sorted(ids))
        & (ds.field("market_start_utc") < cutoff)
        & (ds.field("entry_case") == T60_ENTRY_CASE)
    )
    snapshots = dataset.to_table(columns=T60_COLUMNS, filter=filter_expr).to_pandas()
    snapshots["condition_id"] = snapshots["condition_id"].astype(str).str.lower()
    selected = snapshots[
        snapshots["entry_case"].eq(T60_ENTRY_CASE)
        & snapshots["condition_id"].isin(ids)
    ].copy()
    selected["market_start_utc"] = pd.to_datetime(selected["market_start_utc"], utc=True)
    selected["entry_time_utc"] = pd.to_datetime(selected["entry_time_utc"], utc=True)
    if len(selected) != expected_rows or selected["condition_id"].duplicated().any():
        raise RuntimeError("T-60 snapshot join is not one-to-one on the frozen confirmation subset")
    if not (selected["entry_time_utc"] == selected["market_start_utc"] - pd.Timedelta(seconds=60)).all():
        raise RuntimeError("T-60 source contains an entry time outside the frozen decision cutoff")
    for column in ("fee_known", "has_full_snapshot", "quote_valid", "depth_5usd_valid_both_sides", "no_future_event_at_entry", "no_future_source_event_at_entry"):
        if not selected[column].fillna(False).astype(bool).all():
            raise RuntimeError(f"Frozen T-60 sample fails required source field {column}")
    if selected["bbo_at_entry_ask_mismatches"].fillna(0).ne(0).any():
        raise RuntimeError("Frozen T-60 sample has an ask BBO mismatch")
    return selected.set_index("condition_id", drop=False).to_dict("index")


def _read_t59(ids: set[str], expected_rows: int) -> dict[str, dict]:
    dataset = ds.dataset(T59_PATH, format="parquet")
    cutoff = pa.scalar(DEVELOPMENT_CUTOFF_UTC.to_pydatetime(), type=dataset.schema.field("market_start_utc").type)
    condition_filter = ds.field("condition_id").isin(sorted(ids))
    filter_expr = (
        condition_filter
        & (ds.field("market_start_utc") < cutoff)
        & (ds.field("entry_case") == T59_ENTRY_CASE)
    )
    table = dataset.to_table(columns=T59_COLUMNS, filter=filter_expr)
    quotes = table.to_pandas()
    quotes["condition_id"] = quotes["condition_id"].astype(str).str.lower()
    quotes["market_start_utc"] = pd.to_datetime(quotes["market_start_utc"], utc=True)
    if len(quotes) != expected_rows or quotes["condition_id"].duplicated().any():
        raise RuntimeError("T-59 ask-ladder join is not one-to-one on the frozen confirmation subset")
    for column in T59_REQUIRED_TRUE:
        if not quotes[column].fillna(False).astype(bool).all():
            raise RuntimeError(f"Frozen T-59 sample fails required source field {column}")
    for column in T59_REQUIRED_FALSE:
        if quotes[column].fillna(True).astype(bool).any():
            raise RuntimeError(f"Frozen T-59 sample has an archive continuity gap: {column}")
    if quotes["archive_window_missing_source_hours"].fillna(1).ne(0).any():
        raise RuntimeError("Frozen T-59 sample has missing archive hours")
    if quotes["archive_window_unprocessed_hours"].fillna(1).ne(0).any():
        raise RuntimeError("Frozen T-59 sample has unprocessed archive hours")
    if quotes["bbo_at_entry_ask_mismatches"].fillna(1).ne(0).any():
        raise RuntimeError("Frozen T-59 sample has an ask BBO mismatch")
    return quotes.set_index("condition_id", drop=False).to_dict("index")


def _full_fill(fill: dict, gross_usd: float) -> bool:
    return (
        bool(fill.get("depth_sufficient"))
        and fill.get("fill_status") == "full"
        and fill.get("shares") is not None
        and abs(float(fill.get("gross_usd") or 0.0) - gross_usd) <= FILL_TOLERANCE_USD
    )


def _t60_fill(snapshot: dict, side: str, gross_usd: float, market_start_utc: pd.Timestamp) -> dict | None:
    try:
        price = float(snapshot[f"{side}_best_ask"])
        size = float(snapshot[f"{side}_best_ask_size_shares"])
        fee_rate_bps = float(snapshot["fee_rate_bps"])
    except (TypeError, ValueError, KeyError):
        return None
    if not (0.0 < price < 1.0 and size > 0.0 and price * size >= gross_usd - FILL_TOLERANCE_USD):
        return None
    return economic_replay._walk_asks(
        {price: size},
        fee_rate_bps,
        snapshot["fee_collection_mode"],
        gross_order_usd=gross_usd,
        entry_time_utc=market_start_utc - pd.Timedelta(seconds=59),
        allow_partial=False,
    )


def _official_payout_times(row) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    market_start = pd.Timestamp(row.market_start_utc)
    market_end = market_start + pd.Timedelta(minutes=5)
    outcome_known = pd.Timestamp(row.resolved_at_utc)
    payout_available = max(market_end, outcome_known) + pd.Timedelta(seconds=60)
    return market_end, outcome_known, payout_available


def _run_fixed_five(ledger: pd.DataFrame, model: str) -> tuple[ResolutionPortfolio, dict]:
    account = ResolutionPortfolio(INITIAL_CASH_USD)
    independent_trades = 0
    independent_realized_pnl_usd = 0.0
    for row in ledger.sort_values("market_start_utc").itertuples(index=False):
        market_start = pd.Timestamp(row.market_start_utc)
        decision_time = market_start - pd.Timedelta(seconds=60)
        account.release_payouts_available_by(decision_time)
        side = getattr(row, f"decision_{model}")
        if side == "no_trade":
            continue
        gross = float(getattr(row, f"execution_gross_usd_{model}"))
        if abs(gross - FIXED_GROSS_USD) > FILL_TOLERANCE_USD:
            raise RuntimeError("Saved fixed-control stake differs from the frozen $5 gross amount")
        cash_debit = float(getattr(row, f"execution_cash_debit_usd_{model}"))
        net_shares = float(getattr(row, f"execution_net_shares_{model}"))
        market_end, outcome_known, payout_available = _official_payout_times(row)
        stored_release = pd.Timestamp(getattr(row, f"capital_release_utc_{model}"))
        if payout_available != stored_release:
            raise RuntimeError("Saved E0031 payout availability differs from its frozen account rule")
        expected_payout = net_shares if ((side == "up") == bool(row.target_polymarket_up)) else 0.0
        saved_payout = float(getattr(row, f"settlement_payout_usd_{model}"))
        if abs(expected_payout - saved_payout) > 1e-8:
            raise RuntimeError("Saved E0031 payout does not match official outcome and net shares")
        account.buy(
            position_id=str(row.condition_id),
            side=side,
            gross_usd=gross,
            cash_debit_usd=cash_debit,
            net_shares=net_shares,
            cash_fee_usd=float(getattr(row, f"execution_fee_cash_usd_{model}")),
            share_fee=float(getattr(row, f"execution_fee_shares_{model}")),
            fee_equivalent_usd=float(getattr(row, f"execution_fee_equiv_usd_{model}")),
            official_outcome_up=int(row.target_polymarket_up),
            market_end_utc=market_end,
            outcome_known_at_utc=outcome_known,
            payout_available_at_utc=payout_available,
            entry_time_utc=market_start - pd.Timedelta(seconds=59),
        )
        independent_trades += 1
        independent_realized_pnl_usd += float(getattr(row, f"realized_pnl_usd_{model}"))
    account.release_payouts_available_by(DEVELOPMENT_END_UTC)
    return account, {
        "eligible_markets": len(ledger),
        "signals_with_priceable_fills": independent_trades,
        "independent_realized_pnl_usd": independent_realized_pnl_usd,
    }


def _run_free_cash_five_percent(
    ledger: pd.DataFrame,
    model: str,
    t60_by_id: dict[str, dict],
    t59_by_id: dict[str, dict],
) -> tuple[ResolutionPortfolio, dict]:
    account = ResolutionPortfolio(INITIAL_CASH_USD)
    rejections = {
        "below_historical_minimum_order": 0,
        "no_t60_side_can_fill_requested_notional": 0,
        "no_positive_expected_net_payout": 0,
        "t59_no_full_fill": 0,
        "insufficient_cash_including_fees": 0,
    }
    accepted_sides = {"up": 0, "down": 0}

    for row in ledger.sort_values("market_start_utc").itertuples(index=False):
        market_start = pd.Timestamp(row.market_start_utc)
        decision_time = market_start - pd.Timedelta(seconds=60)
        account.release_payouts_available_by(decision_time)
        requested_gross = FREE_CASH_FRACTION * account.cash_usd
        if requested_gross + FILL_TOLERANCE_USD < FALLBACK_MINIMUM_USD:
            rejections["below_historical_minimum_order"] += 1
            continue

        condition_id = str(row.condition_id).lower()
        t60 = t60_by_id[condition_id]
        t59 = t59_by_id[condition_id]
        p_up = float(getattr(row, f"p_up_{model}"))
        expected_values = {}
        for side, probability in (("up", p_up), ("down", 1.0 - p_up)):
            fill = _t60_fill(t60, side, requested_gross, market_start)
            if fill is not None and _full_fill(fill, requested_gross):
                expected_values[side] = probability * float(fill["shares"]) - float(fill["cash_debit_usd"])
        if not expected_values:
            rejections["no_t60_side_can_fill_requested_notional"] += 1
            continue
        positive_sides = [side for side, value in expected_values.items() if value > 0.0]
        if not positive_sides:
            rejections["no_positive_expected_net_payout"] += 1
            continue
        side = max(positive_sides, key=lambda candidate: (expected_values[candidate], candidate == "up"))

        levels = t59_reassessment._ask_levels(t59[f"{side}_ask_levels"])
        try:
            actual_fill = economic_replay._walk_asks(
                levels,
                t59["fee_rate_bps"],
                t59["fee_collection_mode"],
                gross_order_usd=requested_gross,
                entry_time_utc=market_start - pd.Timedelta(seconds=59),
                allow_partial=False,
            )
        except economic_replay.btc5m_fee_rules.UnknownFeeRuleError:
            actual_fill = None
        if actual_fill is None or not _full_fill(actual_fill, requested_gross):
            rejections["t59_no_full_fill"] += 1
            continue

        market_end, outcome_known, payout_available = _official_payout_times(row)
        accepted = account.buy(
            position_id=condition_id,
            side=side,
            gross_usd=float(actual_fill["gross_usd"]),
            cash_debit_usd=float(actual_fill["cash_debit_usd"]),
            net_shares=float(actual_fill["shares"]),
            cash_fee_usd=float(actual_fill["fee_cash_usd"] or 0.0),
            share_fee=float(actual_fill["fee_shares"] or 0.0),
            fee_equivalent_usd=float(actual_fill["fee_usd"] or 0.0),
            official_outcome_up=int(row.target_polymarket_up),
            market_end_utc=market_end,
            outcome_known_at_utc=outcome_known,
            payout_available_at_utc=payout_available,
            entry_time_utc=market_start - pd.Timedelta(seconds=59),
        )
        if accepted:
            accepted_sides[side] += 1
        else:
            rejections["insufficient_cash_including_fees"] += 1

    account.release_payouts_available_by(DEVELOPMENT_END_UTC)
    return account, {
        "eligible_markets": len(ledger),
        "rejections": rejections,
        "accepted_sides": accepted_sides,
    }


def main() -> None:
    started = time.perf_counter()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["analysis_limits"]["read_holdout_rows"]:
        raise RuntimeError("This runner is frozen to exclude holdout rows")
    input_hashes = verify_protocol_inputs(protocol)

    ledger = pd.read_parquet(LEDGER_PATH)
    _verify_confirmation_ledger(ledger, protocol)
    ledger["condition_id"] = ledger["condition_id"].astype(str).str.lower()
    ids = set(ledger["condition_id"])
    t60_by_id = _read_t60(ids, len(ledger))
    t59_by_id = _read_t59(ids, len(ledger))

    output_dir = OUT_ROOT / f"btc_resolution_development_20261026_{sha256(PROTOCOL_PATH)[:12]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"fixed_5_usd": {}, "free_cash_5pct": {}}
    event_paths = []
    for model in CANDIDATES:
        fixed_account, fixed_counts = _run_fixed_five(ledger, model)
        variable_account, variable_counts = _run_free_cash_five_percent(ledger, model, t60_by_id, t59_by_id)
        results["fixed_5_usd"][model] = {
            **fixed_counts,
            "portfolio": fixed_account.summary(),
        }
        results["free_cash_5pct"][model] = {
            **variable_counts,
            "portfolio": variable_account.summary(),
        }
        for control, account in (("fixed_5_usd", fixed_account), ("free_cash_5pct", variable_account)):
            event_path = output_dir / f"{control}_{model}_events.csv"
            pd.DataFrame(account.events).to_csv(event_path, index=False)
            event_paths.append(event_path)

    observed_timestamps = {
        "market_end_utc": "market_start_utc + 5 minutes",
        "outcome_known_at_utc": "Gamma resolved_at_utc",
        "assumed_payout_available_at_utc": "max(outcome_known_at_utc, market_end_utc) + 60 seconds",
        "actual_redemption_availability": "not archived",
    }
    summary = {
        "experiment_id": protocol["experiment_id"],
        "status": "completed_sparse_confirmation_subset",
        "run_identity_sha256": protocol["source_run_identity_sha256"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "scope": protocol["period"],
        "markets": len(ledger),
        "candidate_identity": protocol["candidate_procedures"],
        "controls": results,
        "settlement_times": observed_timestamps,
        "accounting": {
            "payout_event": "official Gamma resolution, once; no sale or sale fee",
            "locked_equity": "free cash + locked gross cost basis",
            "unresolved_positions_remain_locked": True,
            "independent_trade_diagnostic_source": "unchanged saved E0031 decision ledger, confirmation rows only",
        },
        "limitations": [
            "This is the existing 1,143-market confirmation intersection, not full-development coverage.",
            "T-60 prediction readiness timestamps are absent; saved probabilities are assumed available at the nominal cutoff.",
            "Actual redemption availability timestamps are absent; the reused 60-second release rule is an idealized assumption, not a verified conservative real-world bound.",
            "Gamma official binary outcomes are used for payouts; direct per-market resolution-rule evidence remains incomplete as recorded in D0012/E0033.",
            "Previously exposed confirmation data are not independent validation. The T-59 source file spans into holdout, but only quote/quality columns were projected and the filter returned confirmation IDs before the join; no holdout outcomes or labels were projected or used.",
            "No portfolio-path bootstrap was run; the saved research protocol does not define a valid bootstrap for this stateful cash path.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    run_manifest = {
        "experiment_id": protocol["experiment_id"],
        "status": "completed_sparse_confirmation_subset",
        "protocol_path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_path": Path(__file__).name,
        "runner_sha256": sha256(Path(__file__)),
        "input_sha256": input_hashes,
        "output_sha256": {
            path.name: sha256(path)
            for path in [summary_path, *event_paths]
        },
        "markets": len(ledger),
        "wall_runtime_seconds": time.perf_counter() - started,
        "holdout_rows_used": 0,
        "holdout_outcome_columns_projected": False,
        "t59_confirmation_rows_returned": len(t59_by_id),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": output_dir.relative_to(ROOT).as_posix(),
        "summary": summary["controls"],
        "runtime_seconds": run_manifest["wall_runtime_seconds"],
        "holdout_rows_used": 0,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
