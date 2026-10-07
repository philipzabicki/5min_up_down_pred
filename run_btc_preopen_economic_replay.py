"""Replay BTC pre-open predictions against checkpointed PMXT v2 order-book events."""
from __future__ import annotations

import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import run_btc_preopen_experiment as original_runner


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports/btc_preopen"
MARKET_PATH = ROOT / "data/analysis/polymarket/BTC/new_model_comparison/runs/d794ac2dea5a25a2/shared_market_evaluation.parquet"
INDEX_PATH = ROOT / "data/analysis/polymarket/BTC/preopen_v1/pmxt_v2_archive_full_scenarios/market_index.parquet"
PARTS_DIR = ROOT / "data/analysis/polymarket/BTC/preopen_v1/pmxt_v2_archive_full_scenarios/event_parts"
CANDIDATE_PATH = REPORT_DIR / "candidate_external_predictions.parquet"
SNAPSHOT_CACHE = REPORT_DIR / "entry_snapshots.parquet"
EVENT_SUMMARY_CACHE = REPORT_DIR / "event_replay_summary.json"
REPLAY_CHECKPOINT_PATH = REPORT_DIR / "economic_replay_checkpoint.pkl"
CHECKPOINT_EVERY_PARTS = 12
FEE_EXPONENT_ASSUMPTION = 1.0
FEE_ROUND_DECIMALS = 5
FEE_MINIMUM_USD = 0.00001
LEGACY_FEE_SHARE_DECIMALS = 6
EXCHANGE_PAUSE_START = pd.Timestamp("2026-04-28T11:00:00Z")
EXCHANGE_RESUME = pd.Timestamp("2026-04-28T12:00:00Z")
INITIAL_CASH_USD = 100.0
GROSS_ORDER_USD = 5.0
MAX_ASK_AGE_SECONDS = (1, 5, 30)
SETTLEMENT_RELEASE_DELAYS_SECONDS = (0, 60, 300)
COMPUTE_DELAYS_SECONDS = (0, 15, 45)
MAX_COMPUTE_DELAY_SECONDS = max(COMPUTE_DELAYS_SECONDS)
ORDER_DELAYS_SECONDS = (0, 1, 2, 5)
TRADE_COLUMNS = [
    "entry_case", "entry_kind", "model", "max_ask_age_seconds", "settlement_release_delay_seconds",
    "condition_id", "market_slug", "market_start_utc", "entry_time_utc", "prediction_available_at_utc",
    "resolved_at_utc", "capital_available_again_at_utc", "predicted_up_probability", "chosen_side",
    "chosen_side_expected_net_value_usd", "official_outcome_up", "won", "best_ask", "ask_vwap_5usd",
    "ask_size_at_best_shares", "ask_age_seconds", "quote_source_token_id", "quote_complemented",
    "gross_turnover_usd", "historical_fee_rate_bps", "fee_collection_mode", "fee_exponent_assumption",
    "fees_usd", "gross_shares", "fee_shares", "shares", "cash_debit_usd",
    "cash_available_before_usd", "cash_available_after_entry_usd", "payout_usd", "net_pnl_usd",
]


def _float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _timestamp_ns(values):
    return pd.to_datetime(values, utc=True, errors="coerce").dt.as_unit("ns").astype("int64")


def _levels(value):
    if not isinstance(value, str):
        return {}
    try:
        items = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    result = {}
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        price, size = _float(item[0]), _float(item[1])
        if price is not None and size is not None and 0.0 < price < 1.0 and size > 0.0:
            result[round(price, 8)] = size
    return result


def _book_top(book):
    if not book or not book["bids"] or not book["asks"]:
        return None
    bid = max(book["bids"])
    ask = min(book["asks"])
    return bid, ask, float(book["bids"][bid]), float(book["asks"][ask])


def _fee_collection_mode(entry_time):
    entry_time = pd.Timestamp(entry_time)
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize("UTC")
    else:
        entry_time = entry_time.tz_convert("UTC")
    if EXCHANGE_PAUSE_START <= entry_time < EXCHANGE_RESUME:
        return "maintenance_pause"
    return "outcome_shares" if entry_time < EXCHANGE_PAUSE_START else "cash_collateral"


def _complement(book):
    if not book:
        return None
    return {
        "bids": {round(1.0 - price, 8): size for price, size in book["asks"].items()},
        "asks": {round(1.0 - price, 8): size for price, size in book["bids"].items()},
        "bid_update_ns": book["ask_update_ns"],
        "ask_update_ns": book["bid_update_ns"],
        "bid_source_update_ns": book["ask_source_update_ns"],
        "ask_source_update_ns": book["bid_source_update_ns"],
        "bid_book_update_ns": book["ask_book_update_ns"],
        "ask_book_update_ns": book["bid_book_update_ns"],
        "bid_book_source_ns": book["ask_book_source_ns"],
        "ask_book_source_ns": book["bid_book_source_ns"],
        "source_token_id": book.get("source_token_id", book.get("token_id")),
        "complemented": True,
    }


def _new_state():
    return {
        "tokens": {},
        "event_count": 0,
        "book_count": 0,
        "first_event_ns": None,
        "last_event_ns": None,
        "max_inter_event_gap_seconds": None,
        "uninitialized_price_changes": 0,
        "fee_rate_bps": None,
        "fee_event_ns": None,
        "fee_source_ns": None,
        "bbo_checks": 0,
        "bbo_mismatches": 0,
        "bbo_bid_mismatches": 0,
        "bbo_ask_mismatches": 0,
        "bbo_not_newer_than_side_state": 0,
        "bbo_older_than_side_state": 0,
        "bbo_tied_with_side_state": 0,
        "bbo_not_newer_side_samples": [],
        "bbo_mismatch_samples": [],
        "complement_checks": 0,
        "complement_mismatches": 0,
        "out_of_order_price_changes": 0,
        "out_of_order_book_events": 0,
    }


def _token_state(state, token_id):
    return state["tokens"].setdefault(
        str(token_id),
        {
            "token_id": str(token_id),
            "bids": {},
            "asks": {},
            "initialized": False,
            "book_source_ns": None,
            "last_source_ns": None,
            "book_receive_ns": None,
            "bid_update_ns": None,
            "ask_update_ns": None,
            "bid_source_update_ns": None,
            "ask_source_update_ns": None,
            "bid_book_update_ns": None,
            "ask_book_update_ns": None,
            "bid_book_source_ns": None,
            "ask_book_source_ns": None,
            "book_count": 0,
            "reported_best_bid": None,
            "reported_best_ask": None,
            "reported_source_ns": None,
            "reported_receive_ns": None,
        },
    )


def _direct_book(token):
    if not token or not token["initialized"]:
        return None
    return {
        "bids": token["bids"],
        "asks": token["asks"],
        "bid_update_ns": token["bid_update_ns"],
        "ask_update_ns": token["ask_update_ns"],
        "bid_source_update_ns": token["bid_source_update_ns"],
        "ask_source_update_ns": token["ask_source_update_ns"],
        "bid_book_update_ns": token["bid_book_update_ns"],
        "ask_book_update_ns": token["ask_book_update_ns"],
        "bid_book_source_ns": token["bid_book_source_ns"],
        "ask_book_source_ns": token["ask_book_source_ns"],
        "source_token_id": token["token_id"],
        "complemented": False,
    }


def _effective_book(state, token_id, opposite_id):
    direct = _direct_book(state["tokens"].get(str(token_id)))
    if direct is not None:
        return direct
    return _complement(_direct_book(state["tokens"].get(str(opposite_id))))


def _bbo(book):
    top = _book_top(book)
    return None if top is None else (top[0], top[1])


def _update_event(state, row, received_ns, expected_bbo):
    state["event_count"] += 1
    if state["first_event_ns"] is None:
        state["first_event_ns"] = received_ns
    previous = state["last_event_ns"]
    if previous is not None and received_ns > previous:
        gap = (received_ns - previous) / 1e9
        old_gap = state["max_inter_event_gap_seconds"]
        state["max_inter_event_gap_seconds"] = gap if old_gap is None else max(old_gap, gap)
    state["last_event_ns"] = received_ns

    event_type = row.event_type
    token_id = str(row.asset_id)
    source_ns = int(row.timestamp.value) if not pd.isna(row.timestamp) else received_ns
    if event_type == "last_trade_price":
        fee_rate = _float(row.fee_rate_bps)
        if fee_rate is not None and (
            state.get("fee_source_ns") is None or source_ns >= state["fee_source_ns"]
        ):
            state["fee_rate_bps"] = fee_rate
            state["fee_event_ns"] = received_ns
            state["fee_source_ns"] = source_ns
        return
    if event_type == "book":
        token = _token_state(state, token_id)
        if token["last_source_ns"] is not None and source_ns < token["last_source_ns"]:
            state["out_of_order_book_events"] += 1
            return
        token["bids"] = _levels(row.bids)
        token["asks"] = _levels(row.asks)
        token["initialized"] = bool(token["bids"] and token["asks"])
        token["book_source_ns"] = source_ns
        token["last_source_ns"] = source_ns
        token["book_receive_ns"] = received_ns
        token["bid_update_ns"] = received_ns
        token["ask_update_ns"] = received_ns
        token["bid_source_update_ns"] = source_ns
        token["ask_source_update_ns"] = source_ns
        token["bid_book_update_ns"] = received_ns
        token["ask_book_update_ns"] = received_ns
        token["bid_book_source_ns"] = source_ns
        token["ask_book_source_ns"] = source_ns
        token["book_count"] += 1
        state["book_count"] += 1
        return
    if event_type != "price_change":
        return

    token = _token_state(state, token_id)
    if token["last_source_ns"] is not None and source_ns < token["last_source_ns"]:
        state["out_of_order_price_changes"] += 1
        return
    if token["book_source_ns"] is not None and source_ns <= token["book_source_ns"]:
        return
    bid, ask = _float(row.best_bid), _float(row.best_ask)
    if bid is not None and ask is not None:
        expected_bbo[token_id] = (source_ns, bid, ask)
        token["reported_best_bid"] = bid
        token["reported_best_ask"] = ask
        token["reported_source_ns"] = source_ns
        token["reported_receive_ns"] = received_ns
    if not token["initialized"]:
        token["last_source_ns"] = source_ns
        state["uninitialized_price_changes"] += 1
        return

    prior_top = _book_top(token)
    price, size = _float(row.price), _float(row.size)
    side = str(row.side).upper()
    if price is None or size is None or not 0.0 < price < 1.0 or side not in {"BUY", "SELL"}:
        return
    levels = token["bids"] if side == "BUY" else token["asks"]
    price = round(price, 8)
    old_size = levels.get(price)
    if size <= 0.0:
        levels.pop(price, None)
    else:
        levels[price] = size
    token["last_source_ns"] = source_ns
    changed = (old_size is None) != (size <= 0.0) or (
        old_size is not None and size > 0.0 and old_size != size
    )
    if changed and side == "BUY":
        token["bid_book_update_ns"] = received_ns
        token["bid_book_source_ns"] = source_ns
    elif changed and side == "SELL":
        token["ask_book_update_ns"] = received_ns
        token["ask_book_source_ns"] = source_ns
    current_top = _book_top(token)
    if prior_top is None or current_top is None:
        token["bid_update_ns"] = received_ns
        token["ask_update_ns"] = received_ns
    else:
        if (prior_top[0], prior_top[2]) != (current_top[0], current_top[2]):
            token["bid_update_ns"] = received_ns
            token["bid_source_update_ns"] = source_ns
        if (prior_top[1], prior_top[3]) != (current_top[1], current_top[3]):
            token["ask_update_ns"] = received_ns
            token["ask_source_update_ns"] = source_ns


def _process_receive_group(state, group, received_ns, up_token_id, down_token_id):
    expected = {}
    for row in group.itertuples(index=False):
        _update_event(state, row, received_ns, expected)
    up_book = _effective_book(state, up_token_id, down_token_id)
    down_book = _effective_book(state, down_token_id, up_token_id)
    up_bbo, down_bbo = _bbo(up_book), _bbo(down_book)
    for token_id, (source_ns, expected_bid, expected_ask) in expected.items():
        book = _effective_book(
            state,
            token_id,
            down_token_id if token_id == str(up_token_id) else up_token_id,
        )
        if book is not None and book["source_token_id"] != token_id:
            continue
        source_token = state["tokens"].get(book["source_token_id"]) if book else None
        side_source_timestamps = (
            source_token["bid_book_source_ns"],
            source_token["ask_book_source_ns"],
        ) if source_token is not None else ()
        latest_side_source_ns = (
            max(value for value in side_source_timestamps if value is not None)
            if any(value is not None for value in side_source_timestamps)
            else None
        )
        if latest_side_source_ns is not None and source_ns <= latest_side_source_ns:
            state["bbo_not_newer_than_side_state"] += 1
            if len(state["bbo_not_newer_side_samples"]) < 5:
                actual = _bbo(book)
                state["bbo_not_newer_side_samples"].append({
                    "token_id": token_id,
                    "received_at_utc": pd.Timestamp(received_ns, unit="ns", tz="UTC").isoformat(),
                    "source_timestamp_utc": pd.Timestamp(source_ns, unit="ns", tz="UTC").isoformat(),
                    "latest_side_source_timestamp_utc": pd.Timestamp(
                        latest_side_source_ns, unit="ns", tz="UTC"
                    ).isoformat(),
                    "relation": "older" if source_ns < latest_side_source_ns else "tied",
                    "reported_bbo": [expected_bid, expected_ask],
                    "reconstructed_bbo": None if actual is None else [actual[0], actual[1]],
                })
            if source_ns < latest_side_source_ns:
                state["bbo_older_than_side_state"] += 1
            else:
                state["bbo_tied_with_side_state"] += 1
            continue
        actual = _bbo(book)
        if actual is None:
            continue
        state["bbo_checks"] += 1
        bid_mismatch = abs(actual[0] - expected_bid) > 1e-6
        ask_mismatch = abs(actual[1] - expected_ask) > 1e-6
        if bid_mismatch or ask_mismatch:
            state["bbo_mismatches"] += 1
            state["bbo_bid_mismatches"] += int(bid_mismatch)
            state["bbo_ask_mismatches"] += int(ask_mismatch)
            if len(state["bbo_mismatch_samples"]) < 5:
                state["bbo_mismatch_samples"].append({
                    "token_id": token_id,
                    "received_at_utc": pd.Timestamp(received_ns, unit="ns", tz="UTC").isoformat(),
                    "source_timestamp_utc": pd.Timestamp(source_ns, unit="ns", tz="UTC").isoformat(),
                    "reported_bbo": [expected_bid, expected_ask],
                    "reconstructed_bbo": [actual[0], actual[1]],
                    "source_token_id": book["source_token_id"],
                    "complemented": bool(book["complemented"]),
                    "latest_side_source_timestamp_utc": pd.Timestamp(
                        latest_side_source_ns, unit="ns", tz="UTC"
                    ).isoformat(),
                })
    up_direct = _direct_book(state["tokens"].get(str(up_token_id)))
    down_direct = _direct_book(state["tokens"].get(str(down_token_id)))
    if up_direct is not None and down_direct is not None:
        up_bbo, down_bbo = _bbo(up_direct), _bbo(down_direct)
        if up_bbo is not None and down_bbo is not None:
            state["complement_checks"] += 1
            if abs(up_bbo[0] - (1.0 - down_bbo[1])) > 1e-6 or abs(up_bbo[1] - (1.0 - down_bbo[0])) > 1e-6:
                state["complement_mismatches"] += 1


def _walk_asks(asks, fee_rate_bps, fee_collection_mode):
    remaining = GROSS_ORDER_USD
    gross_shares = 0.0
    raw_fee_cash = 0.0
    raw_fee_shares = 0.0
    fee_usd = 0.0
    gross = 0.0
    for price in sorted(asks):
        available = float(asks[price])
        if not 0.0 < price < 1.0 or available <= 0.0:
            continue
        take = min(remaining / price, available)
        notional = take * price
        gross_shares += take
        gross += notional
        fee_rate = float(fee_rate_bps) / 10_000.0
        if fee_collection_mode == "outcome_shares":
            fee_shares = take * fee_rate * min(price, 1.0 - price) / price
            fee_shares = math.floor(fee_shares * (10 ** LEGACY_FEE_SHARE_DECIMALS) + 1e-9) / (10 ** LEGACY_FEE_SHARE_DECIMALS)
            raw_fee_shares += fee_shares
            fee_usd += fee_shares * price
        elif fee_collection_mode == "cash_collateral":
            raw_fee_cash += take * fee_rate * (price * (1.0 - price)) ** FEE_EXPONENT_ASSUMPTION
        else:
            raise ValueError(f"Unsupported fee collection mode: {fee_collection_mode}")
        remaining -= notional
        if remaining <= 1e-8:
            break
    if remaining > 1e-6:
        return {"depth_sufficient": False, "gross_usd": gross, "gross_shares": gross_shares, "shares": None, "fee_usd": None, "fee_shares": None, "fee_cash_usd": None, "cash_debit_usd": None, "vwap": None}
    if fee_collection_mode == "outcome_shares":
        fee_shares = raw_fee_shares
        fee_cash = 0.0
        net_shares = gross_shares - fee_shares
    else:
        fee_cash = round(raw_fee_cash, FEE_ROUND_DECIMALS)
        if fee_cash < FEE_MINIMUM_USD:
            fee_cash = 0.0
        fee_usd = fee_cash
        fee_shares = 0.0
        net_shares = gross_shares
    return {
        "depth_sufficient": True,
        "gross_usd": gross,
        "gross_shares": gross_shares,
        "shares": net_shares,
        "fee_usd": fee_usd,
        "fee_shares": fee_shares,
        "fee_cash_usd": fee_cash,
        "cash_debit_usd": GROSS_ORDER_USD + fee_cash,
        "vwap": gross / gross_shares if gross_shares > 0.0 else None,
    }


def _valid_reconstructed_quote(bid, ask, ask_size):
    bid_value, ask_value, size_value = _float(bid), _float(ask), _float(ask_size)
    return (
        bid_value is not None
        and ask_value is not None
        and size_value is not None
        and 0.0 < bid_value < 1.0
        and 0.0 < ask_value < 1.0
        and size_value > 0.0
        and bid_value <= ask_value
    )


def _snapshot(state, market, case):
    up_token_id = str(market["up_token_id"])
    down_token_id = str(market["down_token_id"])
    up_book = _effective_book(state, up_token_id, down_token_id)
    down_book = _effective_book(state, down_token_id, up_token_id)
    entry_ns = int(case["entry_time"].value)
    fee_collection_mode = _fee_collection_mode(case["entry_time"])
    rate = state["fee_rate_bps"]
    fee_known = rate is not None and state["fee_event_ns"] is not None and state["fee_event_ns"] <= entry_ns
    fee_age = (entry_ns - state["fee_event_ns"]) / 1e9 if fee_known else None
    sides = {}
    entry_bbo_checks = 0
    entry_bbo_mismatches = 0
    entry_bbo_ask_checks = 0
    entry_bbo_ask_mismatches = 0
    entry_bbo_not_newer_than_side_state = 0
    entry_bbo_older_than_side_state = 0
    entry_bbo_tied_with_side_state = 0
    for outcome, book in (("up", up_book), ("down", down_book)):
        top = _book_top(book)
        ask_update_ns = book["ask_book_source_ns"] if book else None
        ask_receive_ns = book["ask_book_update_ns"] if book else None
        ask_age = (entry_ns - ask_update_ns) / 1e9 if ask_update_ns is not None else None
        ask_receive_age = (entry_ns - ask_receive_ns) / 1e9 if ask_receive_ns is not None else None
        reported_token = state["tokens"].get(up_token_id if outcome == "up" else down_token_id)
        sides[outcome] = {
            "book": book,
            "best_bid": top[0] if top else None,
            "best_ask": top[1] if top else None,
            "best_ask_size_shares": top[3] if top else None,
            "ask_age_seconds": ask_age,
            "ask_received_age_seconds": ask_receive_age,
            "quote_source_token_id": book["source_token_id"] if book else None,
            "quote_complemented": bool(book["complemented"]) if book else False,
            "reported_best_bid": reported_token["reported_best_bid"] if reported_token else None,
            "reported_best_ask": reported_token["reported_best_ask"] if reported_token else None,
            "reported_bbo_receive_ns": reported_token["reported_receive_ns"] if reported_token else None,
            "fill": (
                _walk_asks(book["asks"], rate, fee_collection_mode)
                if book and fee_known and fee_collection_mode != "maintenance_pause"
                else None
            ),
        }
        source_token = state["tokens"].get(book["source_token_id"]) if book else None
        side_state_source_ns = (
            max(
                value
                for value in (
                    source_token["bid_book_source_ns"],
                    source_token["ask_book_source_ns"],
                )
                if value is not None
            )
            if source_token is not None
            and any((
                source_token["bid_book_source_ns"] is not None,
                source_token["ask_book_source_ns"] is not None,
            ))
            else None
        )
        if (
            book is not None
            and top is not None
            and reported_token is not None
            and reported_token["reported_receive_ns"] is not None
            and reported_token["reported_receive_ns"] <= entry_ns
            and source_token is not None
            and book["source_token_id"] == (up_token_id if outcome == "up" else down_token_id)
            and source_token is not None
        ):
            if (
                side_state_source_ns is None
                or reported_token["reported_source_ns"] is None
                or reported_token["reported_source_ns"] <= side_state_source_ns
            ):
                entry_bbo_not_newer_than_side_state += 1
                if (
                    side_state_source_ns is not None
                    and reported_token["reported_source_ns"] is not None
                ):
                    if reported_token["reported_source_ns"] < side_state_source_ns:
                        entry_bbo_older_than_side_state += 1
                    else:
                        entry_bbo_tied_with_side_state += 1
                continue
            entry_bbo_checks += 1
            bid_mismatch = abs(top[0] - reported_token["reported_best_bid"]) > 1e-6
            ask_mismatch = abs(top[1] - reported_token["reported_best_ask"]) > 1e-6
            entry_bbo_ask_checks += 1
            entry_bbo_ask_mismatches += int(ask_mismatch)
            if bid_mismatch or ask_mismatch:
                entry_bbo_mismatches += 1
    direct_book_count = sum(
        bool(token and token["initialized"])
        for token in (state["tokens"].get(up_token_id), state["tokens"].get(down_token_id))
    )
    no_future_source_event = all(
        token["last_source_ns"] is None or token["last_source_ns"] <= entry_ns
        for token in state["tokens"].values()
    ) and (state["fee_source_ns"] is None or state["fee_source_ns"] <= entry_ns)
    bbo_valid = all(
        _valid_reconstructed_quote(
            side["best_bid"],
            side["best_ask"],
            side["best_ask_size_shares"],
        )
        for side in sides.values()
    )
    fill_valid = all(side["fill"] is not None and side["fill"]["depth_sufficient"] for side in sides.values())
    return {
        "condition_id": market["condition_id"],
        "market_slug": market["market_slug"],
        "market_start_utc": market["market_start_utc"],
        "resolved_at_utc": market["resolved_at_utc"],
        "target_polymarket_up": market["target_polymarket_up"],
        "entry_case": case["case_id"],
        "entry_kind": case["kind"],
        "compute_delay_seconds": case["compute_delay_seconds"],
        "order_delay_seconds": case["order_delay_seconds"],
        "entry_time_utc": case["entry_time"],
        "fee_collection_mode": fee_collection_mode,
        "prediction_available_at_utc": case["prediction_available_at"],
        "p_model_raw": market["p_model_raw"],
        "p_model_platt": market["p_model_platt"],
        "p_candidate_raw": market["p_candidate_raw"],
        "p_candidate_platt": market["p_candidate_platt"],
        "book_snapshot_token_count": direct_book_count,
        "has_full_snapshot": direct_book_count == 2,
        "quote_valid": bbo_valid,
        "bbo_at_entry_checks": entry_bbo_checks,
        "bbo_at_entry_mismatches": entry_bbo_mismatches,
        "bbo_at_entry_ask_checks": entry_bbo_ask_checks,
        "bbo_at_entry_ask_mismatches": entry_bbo_ask_mismatches,
        "bbo_at_entry_not_newer_than_side_state": entry_bbo_not_newer_than_side_state,
        "bbo_at_entry_older_than_side_state": entry_bbo_older_than_side_state,
        "bbo_at_entry_tied_with_side_state": entry_bbo_tied_with_side_state,
        "depth_5usd_valid_both_sides": fill_valid,
        "fee_known": fee_known,
        "fee_rate_bps": rate if fee_known else None,
        "fee_age_seconds": fee_age,
        "market_event_count": state["event_count"],
        "market_book_snapshot_count": state["book_count"],
        "market_first_event_utc": pd.Timestamp(state["first_event_ns"], unit="ns", tz="UTC") if state["first_event_ns"] is not None else pd.NaT,
        "market_last_event_utc": pd.Timestamp(state["last_event_ns"], unit="ns", tz="UTC") if state["last_event_ns"] is not None else pd.NaT,
        "no_future_event_at_entry": state["last_event_ns"] is None or state["last_event_ns"] <= entry_ns,
        "no_future_source_event_at_entry": no_future_source_event,
        "max_inter_event_gap_seconds": state["max_inter_event_gap_seconds"],
        "uninitialized_price_change_count": state["uninitialized_price_changes"],
        "bbo_reconciliation_checks": state["bbo_checks"],
        "bbo_reconciliation_mismatches": state["bbo_mismatches"],
        "complement_checks": state["complement_checks"],
        "complement_mismatches": state["complement_mismatches"],
        **{f"{outcome}_{field}": side[field] for outcome, side in sides.items() for field in (
            "best_bid", "best_ask", "best_ask_size_shares", "ask_age_seconds", "ask_received_age_seconds",
            "quote_source_token_id", "quote_complemented", "reported_best_bid", "reported_best_ask",
        )},
        "up_fill": sides["up"]["fill"],
        "down_fill": sides["down"]["fill"],
    }


def _entry_cases(markets):
    schedule = defaultdict(list)
    cases = []
    for market in markets:
        start = market["market_start_utc"]
        decision = start - pd.Timedelta(minutes=1)
        for compute_delay in COMPUTE_DELAYS_SECONDS:
            ready = decision + pd.Timedelta(seconds=compute_delay)
            for order_delay in ORDER_DELAYS_SECONDS:
                case = {
                    "case_id": f"prestart_c{compute_delay}_o{order_delay}",
                    "kind": "prestart",
                    "compute_delay_seconds": compute_delay,
                    "order_delay_seconds": order_delay,
                    "prediction_available_at": ready,
                    "entry_time": ready + pd.Timedelta(seconds=order_delay),
                }
                cases.append(case)
                schedule[int(case["entry_time"].value)].append((market["condition_id"], case))
        ready = decision + pd.Timedelta(seconds=MAX_COMPUTE_DELAY_SECONDS)
        for order_delay in ORDER_DELAYS_SECONDS:
            case = {
                "case_id": f"market_start_c45_o{order_delay}",
                "kind": "market_start_fallback",
                "compute_delay_seconds": MAX_COMPUTE_DELAY_SECONDS,
                "order_delay_seconds": order_delay,
                "prediction_available_at": ready,
                "entry_time": start + pd.Timedelta(seconds=order_delay),
            }
            cases.append(case)
            schedule[int(case["entry_time"].value)].append((market["condition_id"], case))
    return cases, schedule


def _event_latency_stats(arrays):
    if not arrays:
        return {"count": 0, "negative_count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    return {
        "count": int(len(values)),
        "negative_count": int(np.count_nonzero(values < 0)),
        "p50_ms": float(np.quantile(values, 0.50)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "max_ms": float(np.max(values)),
    }


def _accumulate_event_type_counts(counts, part_counts):
    for event_type, count in part_counts.items():
        key = str(event_type)
        counts[key] = int(counts.get(key, 0)) + int(count)


def _extract_snapshots(index):
    markets = index.to_dict("records")
    market_by_id = {market["condition_id"]: market for market in markets}
    cases, schedule = _entry_cases(markets)
    schedule_times = sorted(schedule)
    schedule_position = 0
    snapshots = []
    latency_arrays = []
    event_type_counts = defaultdict(int)
    duplicate_count = 0
    total_events = 0
    part_paths = sorted(PARTS_DIR.glob("*.parquet"))
    part_signature = [
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in part_paths
    ]
    index_signature = (INDEX_PATH.stat().st_size, INDEX_PATH.stat().st_mtime_ns)
    states = defaultdict(_new_state)
    start_position = 0

    if REPLAY_CHECKPOINT_PATH.is_file():
        with REPLAY_CHECKPOINT_PATH.open("rb") as stream:
            checkpoint = pickle.load(stream)
        if (
            checkpoint.get("version") != 3
            or checkpoint.get("part_signature") != part_signature
            or checkpoint.get("index_signature") != index_signature
        ):
            raise RuntimeError("Local PMXT replay checkpoint does not match the current archive and market index")
        states = defaultdict(_new_state, checkpoint["states"])
        schedule_position = int(checkpoint["schedule_position"])
        snapshots = checkpoint["snapshots"]
        latency_arrays = checkpoint["latency_arrays"]
        event_type_counts.update(checkpoint["event_type_counts"])
        duplicate_count = int(checkpoint["duplicate_count"])
        total_events = int(checkpoint["total_events"])
        start_position = int(checkpoint["last_part_position"]) + 1
        print(
            f"[replay] resumed checkpoint after {checkpoint['last_part_name']}: "
            f"snapshots={len(snapshots):,}",
            flush=True,
        )

    def save_checkpoint(part_position, part_name):
        payload = {
            "version": 3,
            "part_signature": part_signature,
            "index_signature": index_signature,
            "last_part_position": part_position,
            "last_part_name": part_name,
            "schedule_position": schedule_position,
            "states": dict(states),
            "snapshots": snapshots,
            "latency_arrays": latency_arrays,
            "event_type_counts": dict(event_type_counts),
            "duplicate_count": duplicate_count,
            "total_events": total_events,
        }
        temporary_path = REPLAY_CHECKPOINT_PATH.with_suffix(".pkl.tmp")
        with temporary_path.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, REPLAY_CHECKPOINT_PATH)
        print(
            f"[replay] saved checkpoint after {part_name}: snapshots={len(snapshots):,}",
            flush=True,
        )

    def emit_before(cutoff_ns, inclusive=False):
        nonlocal schedule_position
        while schedule_position < len(schedule_times):
            at_ns = schedule_times[schedule_position]
            if at_ns > cutoff_ns or (at_ns == cutoff_ns and not inclusive):
                break
            for condition_id, case in schedule[at_ns]:
                market = market_by_id[condition_id]
                snapshots.append(_snapshot(states[condition_id], market, case))
            schedule_position += 1

    for part_position, path in enumerate(part_paths):
        if part_position < start_position:
            continue
        checkpoint_due = (
            (part_position + 1) % CHECKPOINT_EVERY_PARTS == 0
            or part_position + 1 == len(part_paths)
        )
        table = pq.read_table(path)
        if table.num_rows == 0:
            emit_before(int((pd.Timestamp(path.stem.replace("T", " ") + ":00:00", tz="UTC") + pd.Timedelta(hours=1)).value), inclusive=False)
            if checkpoint_due:
                save_checkpoint(part_position, path.name)
            continue
        frame = table.to_pandas()
        frame["market"] = frame["market"].map(lambda value: value.decode("ascii") if isinstance(value, bytes) else str(value))
        frame["_received_ns"] = _timestamp_ns(frame["timestamp_received"])
        frame["_source_ns"] = _timestamp_ns(frame["timestamp"])
        valid_source = frame["_source_ns"].to_numpy(dtype=np.int64, copy=False) > np.iinfo(np.int64).min
        if valid_source.any():
            recv_ms = frame.loc[valid_source, "_received_ns"].to_numpy(dtype=np.int64, copy=False) / 1e6
            source_ms = frame.loc[valid_source, "_source_ns"].to_numpy(dtype=np.int64, copy=False) / 1e6
            latency_arrays.append(recv_ms - source_ms)
        _accumulate_event_type_counts(
            event_type_counts,
            frame["event_type"].value_counts().to_dict(),
        )
        total_events += int(len(frame))
        duplicate_count += int(frame.duplicated(subset=[
            "market", "timestamp_received", "timestamp", "event_type", "asset_id", "price", "size", "side",
        ]).sum())
        frame = frame.loc[frame["market"].isin(market_by_id)]
        if frame.empty:
            hour_end_ns = int((pd.Timestamp(path.stem.replace("T", " ") + ":00:00", tz="UTC") + pd.Timedelta(hours=1)).value)
            emit_before(hour_end_ns, inclusive=False)
            if checkpoint_due:
                save_checkpoint(part_position, path.name)
            continue
        frame.sort_values(["_received_ns", "_source_ns", "market", "asset_id", "event_type"], kind="stable", inplace=True)
        grouped = frame.groupby("_received_ns", sort=False)
        for received_ns, receive_group in grouped:
            emit_before(int(received_ns), inclusive=False)
            receive_group = receive_group.sort_values(["_source_ns", "market", "asset_id", "event_type"], kind="stable")
            for condition_id, event_group in receive_group.groupby("market", sort=False):
                market = market_by_id.get(condition_id)
                if market is None:
                    continue
                _process_receive_group(
                    states[condition_id], event_group,
                    int(received_ns), market["up_token_id"], market["down_token_id"],
                )
            emit_before(int(received_ns), inclusive=True)
        hour_end_ns = int((pd.Timestamp(path.stem.replace("T", " ") + ":00:00", tz="UTC") + pd.Timedelta(hours=1)).value)
        emit_before(hour_end_ns, inclusive=False)
        print(f"[replay] processed {path.name}: events={len(frame):,}, snapshots={len(snapshots):,}", flush=True)
        if checkpoint_due:
            save_checkpoint(part_position, path.name)
    while schedule_position < len(schedule_times):
        emit_before(schedule_times[-1], inclusive=True)
    summary = {
        "event_rows": total_events,
        "exact_duplicate_rows": duplicate_count,
        "event_type_counts": {str(key): int(value) for key, value in event_type_counts.items()},
        "source_receive_latency": _event_latency_stats(latency_arrays),
        "market_states_with_events": len(states),
        "snapshot_rows": len(snapshots),
        "future_event_contaminated_snapshots": int(sum(not snapshot["no_future_event_at_entry"] for snapshot in snapshots)),
        "future_source_timestamp_snapshots": int(sum(not snapshot["no_future_source_event_at_entry"] for snapshot in snapshots)),
        "book_snapshot_events": int(sum(state["book_count"] for state in states.values())),
        "bbo_reconciliation_checks": int(sum(state["bbo_checks"] for state in states.values())),
        "bbo_reconciliation_mismatches": int(sum(state["bbo_mismatches"] for state in states.values())),
        "bbo_references_not_newer_than_side_state": int(sum(state["bbo_not_newer_than_side_state"] for state in states.values())),
        "bbo_references_older_than_side_state": int(sum(state["bbo_older_than_side_state"] for state in states.values())),
        "bbo_references_tied_with_side_state": int(sum(state["bbo_tied_with_side_state"] for state in states.values())),
        "bbo_bid_reconciliation_mismatches": int(sum(state["bbo_bid_mismatches"] for state in states.values())),
        "bbo_ask_reconciliation_mismatches": int(sum(state["bbo_ask_mismatches"] for state in states.values())),
        "complement_checks": int(sum(state["complement_checks"] for state in states.values())),
        "complement_mismatches": int(sum(state["complement_mismatches"] for state in states.values())),
        "out_of_order_price_changes": int(sum(state["out_of_order_price_changes"] for state in states.values())),
        "out_of_order_book_events": int(sum(state["out_of_order_book_events"] for state in states.values())),
        "bbo_reconciliation_first_mismatch_samples": [sample for state in states.values() for sample in state["bbo_mismatch_samples"][:5]][:20],
        "bbo_references_not_newer_samples": [sample for state in states.values() for sample in state["bbo_not_newer_side_samples"][:5]][:20],
        "markets_with_uninitialized_deltas": int(sum(state["uninitialized_price_changes"] > 0 for state in states.values())),
        "max_market_inter_event_gap_seconds": max((state["max_inter_event_gap_seconds"] or 0.0 for state in states.values()), default=0.0),
    }
    return pd.DataFrame(snapshots), summary


def _count_cached_event_types(part_paths):
    counts = {}
    for path in part_paths:
        table = pq.read_table(path, columns=["event_type"])
        part_counts = {
            row["values"]: int(row["counts"])
            for row in table.column("event_type").value_counts().to_pylist()
        }
        _accumulate_event_type_counts(
            counts,
            part_counts,
        )
    return counts


def _data_reason(row, max_age_seconds):
    if pd.isna(row.target_polymarket_up) or pd.isna(row.resolved_at_utc):
        return "missing_official_resolution"
    if row.fee_collection_mode == "maintenance_pause":
        return "exchange_maintenance_pause"
    if not row.no_future_event_at_entry:
        return "future_event_contamination"
    if not row.no_future_source_event_at_entry:
        return "future_source_timestamp_at_entry"
    if not row.has_full_snapshot:
        return "no_initial_book_snapshot"
    if not row.quote_valid:
        return "incomplete_or_crossed_book"
    if row.bbo_at_entry_ask_mismatches:
        return "unreconciled_best_ask_at_entry"
    if row.up_ask_age_seconds is None or row.down_ask_age_seconds is None:
        return "missing_ask_timestamp"
    if max(row.up_ask_age_seconds, row.down_ask_age_seconds) > max_age_seconds:
        return "stale_ask"
    if not row.fee_known:
        return "unknown_historical_fee"
    if not row.depth_5usd_valid_both_sides:
        return "insufficient_5usd_ask_depth"
    return "eligible"


def _probability_ev(probability, fill):
    if not fill or not fill["depth_sufficient"] or fill["fee_usd"] is None:
        return None
    return float(probability) * fill["shares"] - fill["cash_debit_usd"]


def _simulate_group(group, model_name, max_age_seconds, release_delay_seconds, keep_trades):
    probability_col = {
        "original_v1_raw": "p_model_raw",
        "original_v1_platt": "p_model_platt",
        "candidate_raw": "p_candidate_raw",
        "candidate_platt": "p_candidate_platt",
    }[model_name]
    ordered = group.sort_values("entry_time_utc", kind="stable")
    cash = INITIAL_CASH_USD
    locked_cost = 0.0
    open_positions = []
    trades = []
    equity_peak = INITIAL_CASH_USD
    max_drawdown = 0.0
    max_locked_cost = 0.0
    counters = defaultdict(int)
    fees_paid = turnover = payout_total = 0.0
    realized_pnl = 0.0
    trade_count = 0

    def equity_now():
        return cash + locked_cost

    def record_equity():
        nonlocal equity_peak, max_drawdown, max_locked_cost
        max_locked_cost = max(max_locked_cost, locked_cost)
        equity = equity_now()
        equity_peak = max(equity_peak, equity)
        if equity_peak > 0:
            max_drawdown = max(max_drawdown, (equity_peak - equity) / equity_peak)

    def settle_until(timestamp):
        nonlocal cash, locked_cost, payout_total, realized_pnl
        open_positions.sort(key=lambda item: item[0])
        while open_positions and open_positions[0][0] <= timestamp:
            release_ns, payload = open_positions.pop(0)
            locked_cost -= payload["gross_usd"]
            cash += payload["payout_usd"]
            payout_total += payload["payout_usd"]
            realized_pnl += payload["pnl_net_usd"]
            record_equity()

    for row in ordered.itertuples(index=False):
        entry_ns = int(pd.Timestamp(row.entry_time_utc).value)
        settle_until(entry_ns)
        counters["markets_seen"] += 1
        reason = _data_reason(row, max_age_seconds)
        if reason != "eligible":
            counters[f"reject_{reason}"] += 1
            continue
        p_up = _float(getattr(row, probability_col))
        if p_up is None or not 0.0 <= p_up <= 1.0:
            counters["reject_missing_probability"] += 1
            continue
        up_fill, down_fill = row.up_fill, row.down_fill
        up_ev = _probability_ev(p_up, up_fill)
        down_ev = _probability_ev(1.0 - p_up, down_fill)
        if up_ev is None or down_ev is None:
            counters["reject_insufficient_5usd_ask_depth"] += 1
            continue
        if max(up_ev, down_ev) <= 0.0:
            counters["skip_no_positive_expected_edge"] += 1
            continue
        side = "up" if up_ev >= down_ev else "down"
        fill = up_fill if side == "up" else down_fill
        ev = max(up_ev, down_ev)
        debit = float(fill["cash_debit_usd"])
        if cash + 1e-9 < debit:
            counters["reject_insufficient_balance"] += 1
            continue
        outcome = int(row.target_polymarket_up)
        won = outcome == (1 if side == "up" else 0)
        payout = float(fill["shares"]) if won else 0.0
        pnl_net = payout - debit
        release_at = pd.Timestamp(row.resolved_at_utc) + pd.Timedelta(seconds=release_delay_seconds)
        cash_before = cash
        cash -= debit
        locked_cost += GROSS_ORDER_USD
        fees_paid += float(fill["fee_usd"])
        turnover += GROSS_ORDER_USD
        trade_count += 1
        counters["trades"] += 1
        open_positions.append((int(release_at.value), {
            "gross_usd": GROSS_ORDER_USD,
            "payout_usd": payout,
            "pnl_net_usd": pnl_net,
        }))
        record_equity()
        if keep_trades:
            trades.append({
                "entry_case": row.entry_case,
                "entry_kind": row.entry_kind,
                "model": model_name,
                "max_ask_age_seconds": max_age_seconds,
                "settlement_release_delay_seconds": release_delay_seconds,
                "condition_id": row.condition_id,
                "market_slug": row.market_slug,
                "market_start_utc": row.market_start_utc,
                "entry_time_utc": row.entry_time_utc,
                "prediction_available_at_utc": row.prediction_available_at_utc,
                "resolved_at_utc": row.resolved_at_utc,
                "capital_available_again_at_utc": release_at,
                "predicted_up_probability": p_up,
                "chosen_side": side,
                "chosen_side_expected_net_value_usd": ev,
                "official_outcome_up": outcome,
                "won": won,
                "best_ask": row.up_best_ask if side == "up" else row.down_best_ask,
                "ask_vwap_5usd": fill["vwap"],
                "ask_size_at_best_shares": row.up_best_ask_size_shares if side == "up" else row.down_best_ask_size_shares,
                "ask_age_seconds": row.up_ask_age_seconds if side == "up" else row.down_ask_age_seconds,
                "quote_source_token_id": row.up_quote_source_token_id if side == "up" else row.down_quote_source_token_id,
                "quote_complemented": row.up_quote_complemented if side == "up" else row.down_quote_complemented,
                "gross_turnover_usd": GROSS_ORDER_USD,
                "historical_fee_rate_bps": row.fee_rate_bps,
                "fee_collection_mode": row.fee_collection_mode,
                "fee_exponent_assumption": FEE_EXPONENT_ASSUMPTION,
                "fees_usd": fill["fee_usd"],
                "gross_shares": fill["gross_shares"],
                "fee_shares": fill["fee_shares"],
                "shares": fill["shares"],
                "cash_debit_usd": debit,
                "cash_available_before_usd": cash_before,
                "cash_available_after_entry_usd": cash,
                "payout_usd": payout,
                "net_pnl_usd": pnl_net,
            })
    settle_until(2**63 - 1)
    counters["data_rejections"] = int(sum(value for key, value in counters.items() if key.startswith("reject_") and key != "reject_insufficient_balance"))
    total_pnl = cash - INITIAL_CASH_USD
    result = {
        "scenario_id": f"{model_name}|{group.entry_case.iloc[0]}|age{max_age_seconds}|release{release_delay_seconds}",
        "model": model_name,
        "entry_case": group.entry_case.iloc[0],
        "entry_kind": group.entry_kind.iloc[0],
        "compute_delay_seconds": int(group.compute_delay_seconds.iloc[0]),
        "order_delay_seconds": int(group.order_delay_seconds.iloc[0]),
        "max_ask_age_seconds": max_age_seconds,
        "settlement_release_delay_seconds": release_delay_seconds,
        "initial_cash_usd": INITIAL_CASH_USD,
        "ending_cash_usd": cash,
        "net_pnl_usd": total_pnl,
        "gross_turnover_usd": turnover,
        "roi_on_initial_cash": total_pnl / INITIAL_CASH_USD,
        "roi_on_gross_turnover": total_pnl / turnover if turnover > 0.0 else None,
        "fees_paid_usd": fees_paid,
        "total_payout_usd": payout_total,
        "max_drawdown_at_cost": max_drawdown,
        "trade_count": trade_count,
        "max_open_cost_basis_usd": max_locked_cost,
        **dict(counters),
    }
    return result, trades


def _predictive_metrics_and_intervals(markets):
    prevalence = 0.5015364895
    rows_out = []
    interval_details = {}
    for target_name, target_col in (
        ("binance_proxy", "target_binance_proxy_up"),
        ("official_polymarket", "target_polymarket_up"),
    ):
        data = markets.loc[markets[target_col].notna()].copy()
        y = data[target_col].astype(np.int8).to_numpy()
        probs = {
            "original_v1_raw": data["p_model_raw"].to_numpy(dtype=float),
            "original_v1_platt": data["p_model_platt"].to_numpy(dtype=float),
            "candidate_raw": data["p_candidate_raw"].to_numpy(dtype=float),
            "candidate_platt": data["p_candidate_platt"].to_numpy(dtype=float),
        }
        ci = {}
        baseline_half = np.full(len(y), 0.5, dtype=float)
        baseline_prevalence = np.full(len(y), prevalence, dtype=float)
        for model_name, probability in probs.items():
            metric = {
                "n": int(len(y)),
                "positive_rate": float(y.mean()),
                "logloss": float(log_loss(y, probability, labels=[0, 1])),
                "brier": float(brier_score_loss(y, probability)),
                "auc": float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else None,
            }
            half_ci = original_runner._paired_block_interval(y, probability, baseline_half)
            prevalence_ci = original_runner._paired_block_interval(y, probability, baseline_prevalence)
            ci[model_name] = {"vs_constant_0_5": half_ci, "vs_development_prevalence_0_5015364895": prevalence_ci}
            rows_out.append({
                "candidate": model_name,
                "scope": f"external {target_name}",
                **metric,
                "development_prevalence_baseline": prevalence,
                "constant_0_5_logloss": math.log(2.0),
                "constant_0_5_brier": 0.25,
                "development_prevalence_logloss": float(log_loss(y, baseline_prevalence, labels=[0, 1])),
                "development_prevalence_brier": float(brier_score_loss(y, baseline_prevalence)),
                "vs_0_5_delta_logloss_ci95": json.dumps(half_ci["delta_first_minus_second_log_loss_ci95"]),
                "vs_0_5_delta_brier_ci95": json.dumps(half_ci["delta_first_minus_second_brier_ci95"]),
                "vs_development_prevalence_delta_logloss_ci95": json.dumps(prevalence_ci["delta_first_minus_second_log_loss_ci95"]),
                "vs_development_prevalence_delta_brier_ci95": json.dumps(prevalence_ci["delta_first_minus_second_brier_ci95"]),
            })
        paired = {}
        for first, second in (("candidate_raw", "original_v1_raw"), ("candidate_platt", "original_v1_platt"), ("original_v1_platt", "original_v1_raw"), ("candidate_platt", "candidate_raw")):
            paired[f"{first}_minus_{second}"] = original_runner._paired_block_interval(y, probs[first], probs[second])
        interval_details[target_name] = {"n": int(len(y)), "positive_rate": float(y.mean()), "paired_intervals": paired, "model_vs_baseline_intervals": ci}
    return rows_out, interval_details


def run_replay():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    index = pd.read_parquet(INDEX_PATH)
    for column in ("market_start_utc", "resolved_at_utc"):
        index[column] = pd.to_datetime(index[column], utc=True, errors="coerce")
    candidate = pd.read_parquet(CANDIDATE_PATH)
    candidate["market_start_utc"] = pd.to_datetime(candidate["market_start_utc"], utc=True)
    candidate = candidate.drop_duplicates("condition_id")
    candidate["Opened"] = pd.to_datetime(candidate["Opened"], utc=True)
    original_predictions = pd.read_parquet(
        ROOT / "data/analysis/polymarket/BTC/preopen_v1/nightly_v1/06dc69ccb35a63b3/stages/external_evaluation_7e5807168d4c/external_test_predictions.parquet",
        columns=["Opened", "p_model_raw", "p_model_platt"],
    )
    original_predictions["Opened"] = pd.to_datetime(original_predictions["Opened"], utc=True)
    full_predictions = candidate.merge(
        original_predictions, on="Opened", how="inner", validate="one_to_one",
    )
    if len(full_predictions) != len(candidate):
        raise RuntimeError("Candidate and original external prediction rows do not align by decision time")
    prediction_rows, predictive_intervals = _predictive_metrics_and_intervals(full_predictions)
    index = index.merge(
        candidate[["condition_id", "p_candidate_raw", "p_candidate_platt"]],
        on="condition_id", how="left", validate="one_to_one",
    )
    original = pd.read_parquet(MARKET_PATH, columns=[
        "condition_id", "target_polymarket_up", "target_binance_proxy_up", "p_model_up", "new_btc_platt",
    ])
    original.rename(columns={"p_model_up": "p_model_raw", "new_btc_platt": "p_model_platt"}, inplace=True)
    index = index.merge(
        original, on="condition_id", how="left", validate="one_to_one", suffixes=("", "_source"),
    )
    expected_snapshot_rows = int(len(index) * 16)
    cached_summary = (
        json.loads(EVENT_SUMMARY_CACHE.read_text(encoding="utf-8"))
        if EVENT_SUMMARY_CACHE.is_file()
        else {}
    )
    reused_snapshot_cache = (
        SNAPSHOT_CACHE.is_file()
        and cached_summary.get("book_replay_semantics_version") == 3
        and cached_summary.get("bbo_references_not_newer_than_side_state") is not None
    )
    if reused_snapshot_cache:
        snapshots = pd.read_parquet(SNAPSHOT_CACHE)
        event_summary = json.loads(EVENT_SUMMARY_CACHE.read_text(encoding="utf-8"))
        print(f"[replay] reused {len(snapshots):,} cached full-depth snapshots", flush=True)
    else:
        snapshots, event_summary = _extract_snapshots(index)
        event_summary["book_replay_semantics_version"] = 3
    if snapshots.empty:
        raise RuntimeError("No PMXT entry snapshots were produced")
    if len(snapshots) != expected_snapshot_rows:
        raise RuntimeError(f"Expected {expected_snapshot_rows} market-entry snapshots; received {len(snapshots)}")
    if not snapshots["no_future_event_at_entry"].all():
        raise RuntimeError("At least one entry snapshot includes a future-received PMXT event")
    if not reused_snapshot_cache:
        snapshots.to_parquet(SNAPSHOT_CACHE, index=False)
    event_summary["event_type_counts"] = _count_cached_event_types(PARTS_DIR.glob("*.parquet"))
    if not reused_snapshot_cache:
        EVENT_SUMMARY_CACHE.write_text(json.dumps(event_summary, indent=2, default=str) + "\n", encoding="utf-8")
        REPLAY_CHECKPOINT_PATH.unlink(missing_ok=True)
        REPLAY_CHECKPOINT_PATH.with_suffix(".pkl.tmp").unlink(missing_ok=True)
    else:
        EVENT_SUMMARY_CACHE.write_text(json.dumps(event_summary, indent=2, default=str) + "\n", encoding="utf-8")
    snapshots["entry_time_utc"] = pd.to_datetime(snapshots["entry_time_utc"], utc=True)
    snapshots["resolved_at_utc"] = pd.to_datetime(snapshots["resolved_at_utc"], utc=True, errors="coerce")
    snapshots["quote_valid_strict_bid_lt_ask"] = snapshots.apply(
        lambda row: all(
            _float(row[f"{outcome}_best_bid"]) is not None
            and _float(row[f"{outcome}_best_ask"]) is not None
            and _float(row[f"{outcome}_best_bid"]) < _float(row[f"{outcome}_best_ask"])
            for outcome in ("up", "down")
        ),
        axis=1,
    )
    snapshots["quote_valid"] = snapshots.apply(
        lambda row: all(
            _valid_reconstructed_quote(
                row[f"{outcome}_best_bid"],
                row[f"{outcome}_best_ask"],
                row[f"{outcome}_best_ask_size_shares"],
            )
            for outcome in ("up", "down")
        ),
        axis=1,
    )
    snapshots.to_parquet(SNAPSHOT_CACHE, index=False)
    for outcome in ("up", "down"):
        snapshots[f"{outcome}_ask_age_seconds"] = pd.to_numeric(snapshots[f"{outcome}_ask_age_seconds"], errors="coerce")
    coverage_columns = [
        "condition_id", "market_slug", "market_start_utc", "entry_case", "entry_kind",
        "compute_delay_seconds", "order_delay_seconds", "prediction_available_at_utc", "entry_time_utc",
        "target_polymarket_up", "no_future_event_at_entry", "no_future_source_event_at_entry", "has_full_snapshot", "book_snapshot_token_count", "quote_valid",
        "quote_valid_strict_bid_lt_ask",
        "fee_collection_mode",
        "bbo_at_entry_checks", "bbo_at_entry_mismatches",
        "bbo_at_entry_ask_checks", "bbo_at_entry_ask_mismatches",
        "bbo_at_entry_not_newer_than_side_state",
        "bbo_at_entry_older_than_side_state", "bbo_at_entry_tied_with_side_state",
        "depth_5usd_valid_both_sides", "fee_known", "fee_rate_bps", "fee_age_seconds",
        "up_best_bid", "up_best_ask", "up_reported_best_bid", "up_reported_best_ask", "up_best_ask_size_shares", "up_ask_age_seconds", "up_ask_received_age_seconds", "up_quote_complemented",
        "down_best_bid", "down_best_ask", "down_reported_best_bid", "down_reported_best_ask", "down_best_ask_size_shares", "down_ask_age_seconds", "down_ask_received_age_seconds", "down_quote_complemented",
        "market_event_count", "market_book_snapshot_count", "market_first_event_utc", "market_last_event_utc",
        "max_inter_event_gap_seconds", "uninitialized_price_change_count", "bbo_reconciliation_checks",
        "bbo_reconciliation_mismatches", "complement_checks", "complement_mismatches",
    ]
    for age in MAX_ASK_AGE_SECONDS:
        snapshots[f"coverage_reason_age{age}s"] = snapshots.apply(lambda row: _data_reason(row, age), axis=1)
    snapshots[coverage_columns + [f"coverage_reason_age{age}s" for age in MAX_ASK_AGE_SECONDS]].to_csv(
        REPORT_DIR / "data_coverage.csv", index=False,
    )

    economy_rows, trade_rows = [], []
    for entry_case, group in snapshots.groupby("entry_case", sort=False):
        for model in ("original_v1_raw", "original_v1_platt", "candidate_raw", "candidate_platt"):
            for max_age in MAX_ASK_AGE_SECONDS:
                for release_delay in SETTLEMENT_RELEASE_DELAYS_SECONDS:
                    result, trades = _simulate_group(
                        group, model, max_age, release_delay,
                        keep_trades=(max_age == 30),
                    )
                    economy_rows.append(result)
                    trade_rows.extend(trades)
    economic = pd.DataFrame(economy_rows)
    economic.to_csv(REPORT_DIR / "economic_scenarios.csv", index=False)
    pd.DataFrame(trade_rows, columns=TRADE_COLUMNS).to_parquet(REPORT_DIR / "trades.parquet", index=False)
    pd.DataFrame(prediction_rows).to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    economic_summary = {
        "money_and_execution": {
            "initial_cash_usd": INITIAL_CASH_USD,
            "target_gross_order_usd": GROSS_ORDER_USD,
            "fee_in_cash_debit": "gross $5 order plus any collateral-denominated fee; historical share-denominated fees reduce the payout shares instead",
            "order_minimum_shares": "historical minimum not available; no current minimum was projected backward",
            "ev_rule": "choose UP or DOWN with the larger positive expected net payout after ask depth and historical fee; otherwise skip",
            "execution_price": "walk ask-side full depth to $5 gross; no midpoint or future quote",
            "event_availability_clock": "timestamp_received",
            "quote_freshness_age_clock": "source timestamp of the latest changed ask-side level; received timestamp gates availability and its age is retained separately",
            "fee_rate": "latest fee_rate_bps last_trade_price event received at/before entry, divided by 10000",
            "fee_collection_mode": "outcome shares before 2026-04-28 11:00 UTC; entries from 11:00 to 12:00 UTC excluded as exchange maintenance; collateral cash from 12:00 UTC onward",
            "legacy_fee_curve": "fee shares = shares * rate * min(price, 1-price) / price; aggregate depth levels rounded down to 6 share decimals; PMXT does not expose maker-level fills/rounding",
            "collateral_fee_curve": "cash fee = shares * rate * (price * (1-price))^1, exponent 1; historical exponent is not included in PMXT rows",
            "collateral_fee_rounding": {"decimals": FEE_ROUND_DECIMALS, "minimum_fee_usd": FEE_MINIMUM_USD},
            "maker_fill_fee_precision_limit": "book depth is aggregated; the archive has no maker-level matches. The replay rounds the cash fee for the aggregate order and share fees per aggregate price level, so exact per-match rounding cannot be recovered",
            "fee_accounting": "fees_paid_usd reports the collateral fee or legacy share-fee value at execution price; cash_debit_usd adds fees only when historically charged in collateral",
            "quote_freshness_scenarios_seconds": list(MAX_ASK_AGE_SECONDS),
            "compute_delay_scenarios_seconds": list(COMPUTE_DELAYS_SECONDS),
            "order_delay_scenarios_seconds": list(ORDER_DELAYS_SECONDS),
            "settlement_release_delay_scenarios_seconds": list(SETTLEMENT_RELEASE_DELAYS_SECONDS),
            "settlement_cash_policy": "cash plus cost basis of locked positions until official resolved_at_utc and the scenario release delay; no credit or deposits",
            "drawdown": "peak-to-trough of available cash plus locked shares at entry cost; excludes intratrade mark-to-market",
        },
        "event_archive": event_summary,
        "predictive_intervals": predictive_intervals,
        "primary_scenario": "prestart_c0_o1|age30|release60 (historical cycle-complete p50 rounded upward to a one-second entry delay; entry T-59s)",
        "main_entry_times_relative_to_market_start": {
            "prestart_c0_o1": "T-59s (historical cycle-complete p50 of 475.28ms rounded upward to one second)",
            "prestart_c0_o0": "T-60s (ideal reference)",
            "prestart_c0_o2": "T-58s (+2s sensitivity)",
            "prestart_c0_o5": "T-55s (conservative local-host component scenario)",
        },
        "robustness_scenarios": {
            "prestart_c15_o5": "prediction ready T-45s; entry T-40s",
            "prestart_c45_o5": "prediction ready T-15s; entry T-10s",
        },
        "markets": int(index.condition_id.nunique()),
        "markets_with_official_labels": int(index.target_polymarket_up.notna().sum()),
        "markets_with_candidate_probabilities": int(index.p_candidate_raw.notna().sum()),
        "market_start_fallback_is_separate": True,
    }
    (REPORT_DIR / "economic_replay.json").write_text(json.dumps(economic_summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "event_summary": event_summary,
        "coverage_rows": len(snapshots),
        "economic_scenario_rows": len(economic),
        "trade_rows": len(trade_rows),
        "primary": economic.loc[
            economic.entry_case.eq("prestart_c0_o1")
            & economic.max_ask_age_seconds.eq(30)
            & economic.settlement_release_delay_seconds.eq(60)
        ].to_dict("records"),
    }, indent=2, default=str), flush=True)


if __name__ == "__main__":
    run_replay()
