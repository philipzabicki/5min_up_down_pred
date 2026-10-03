"""Reproduce BTC Platt entry diagnostics without portfolio cash constraints."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils.polymarket_market_value import _execution_quotes, simulate_fixed_policy


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "data/analysis/polymarket/BTC/new_model_comparison/runs/d794ac2dea5a25a2"
QUOTE_PATH = ROOT / "data/datasets/polymarket/BTC/runs/efa1bf384fc6a2dc/quotes.parquet"
OUTPUT_DIR = ROOT / "data/analysis/polymarket/BTC/entry_selection_audit_20261003"
STAKE_USD = 5.0
BANKROLL_USD = 100.0
MIN_EXPECTED_PNL_USD = 0.25
EXECUTION_DELAY_SECONDS = 1
PRICE_BINS = (0.0, 0.35, 0.45, 1.01)
PRICE_LABELS = ("<0.35", "0.35-0.45", ">=0.45")
GAP_BINS = (-2.0, 0.05, 0.10, 2.0)
GAP_LABELS = ("<0.05", "0.05-0.10", ">=0.10")


def _native(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _score_rows(frame, group_columns):
    grouped = frame.groupby(group_columns, observed=True, dropna=False)
    result = grouped.agg(
        n=("condition_id", "size"),
        mean_model_probability=("p_model_side", "mean"),
        mean_market_probability=("p_market_side", "mean"),
        actual_win_rate=("won", "mean"),
        mean_execution_price=("execution_price", "mean"),
        mean_break_even_probability=("break_even_probability_after_cost", "mean"),
        mean_expected_net_pnl_usd=("expected_net_pnl_usd", "mean"),
        pnl_after_cost_usd=("pnl", "sum"),
    )
    return result.reset_index()


def main():
    started = time.perf_counter()
    evaluation = pd.read_parquet(RUN_DIR / "shared_market_evaluation.parquet")
    portfolio_frame = evaluation.copy()
    portfolio_frame["p_model_up"] = portfolio_frame["oof_raw"]

    future = _execution_quotes(
        portfolio_frame,
        QUOTE_PATH,
        EXECUTION_DELAY_SECONDS,
    ).set_index("condition_id")
    replay_kwargs = {
        "execution_delay": EXECUTION_DELAY_SECONDS,
        "future_quotes": future,
        "min_expected_pnl": MIN_EXPECTED_PNL_USD,
    }
    independent_summary, independent_trades, _ = simulate_fixed_policy(
        portfolio_frame,
        "new_btc_platt",
        bankroll=None,
        **replay_kwargs,
    )
    portfolio_summary, portfolio_trades, _ = simulate_fixed_policy(
        portfolio_frame,
        "new_btc_platt",
        bankroll=BANKROLL_USD,
        **replay_kwargs,
    )

    if independent_trades.condition_id.duplicated().any():
        raise AssertionError("Independent diagnostics contain duplicate market entries")
    portfolio_ids = set(portfolio_trades.condition_id)
    all_ids = set(independent_trades.condition_id)
    if not portfolio_ids <= all_ids:
        raise AssertionError("Portfolio entries are not a subset of independent signals")
    cash_rejected = int(portfolio_summary["rejection_reasons"].get("insufficient_cash", 0))
    if cash_rejected != len(all_ids - portfolio_ids):
        raise AssertionError("Cash rejections do not reconcile to independent signals")

    source = portfolio_frame
    quote = future.reindex(independent_trades.condition_id).reset_index()
    signals = independent_trades.merge(
        source[[
            "condition_id", "Opened", "market_start_utc", "decision_available_at",
            "market_end_utc", "resolved_at_utc", "timestamp_utc", "fee_rate",
            "fee_exponent", "fee_round_decimals", "fee_min_fee", "order_min_size",
        ]],
        on="condition_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_source"),
    ).merge(
        quote[[
            "condition_id", "quote_valid", "quote_delay_ms",
            "book_age_known", "stale_observation", "up_best_bid_quote",
            "up_best_ask_quote", "down_best_bid_quote", "down_best_ask_quote",
        ]],
        on="condition_id",
        how="left",
        validate="one_to_one",
    )
    if signals["target_polymarket_up"].isna().any() or signals["resolved_at_utc"].isna().any():
        raise AssertionError("Official outcome or resolution timestamp is missing")

    signals["p_model_side"] = signals["p_success"]
    signals["p_market_side"] = signals["p_market_mid"].where(
        signals["side"].eq("up"), 1.0 - signals["p_market_mid"]
    )
    signals["model_market_probability_gap"] = (
        signals["p_model_side"] - signals["p_market_side"]
    )
    signals["break_even_probability_after_cost"] = signals["stake"] / signals["shares"]
    signals["predicted_net_edge_probability"] = (
        signals["p_model_side"] - signals["break_even_probability_after_cost"]
    )
    signals["expected_net_pnl_usd"] = (
        signals["p_model_side"] * signals["shares"] - signals["stake"]
    )
    signals["won"] = signals["pnl"].gt(0)
    signals["in_100_usd_portfolio_replay"] = signals["condition_id"].isin(portfolio_ids)
    signals["portfolio_rejection_reason"] = np.where(
        signals["in_100_usd_portfolio_replay"], "executed_in_replay", "insufficient_cash"
    )
    signals["first_quote_delay_after_prediction_seconds"] = (
        signals["timestamp_utc"] - signals["decision_available_at"]
    ).dt.total_seconds()
    signals["execution_quote_delay_after_prediction_seconds"] = (
        signals["execution_quote_at"] - signals["decision_available_at"]
    ).dt.total_seconds()
    signals["execution_quote_delay_after_market_start_seconds"] = (
        signals["execution_quote_at"] - signals["market_start_utc"]
    ).dt.total_seconds()
    signals["market_mid_side_at_execution"] = np.where(
        signals["side"].eq("up"),
        (signals["up_best_bid_quote"] + signals["up_best_ask_quote"]) / 2.0,
        (signals["down_best_bid_quote"] + signals["down_best_ask_quote"]) / 2.0,
    )
    up_mid_execution = (signals["up_best_bid_quote"] + signals["up_best_ask_quote"]) / 2.0
    down_mid_execution = (signals["down_best_bid_quote"] + signals["down_best_ask_quote"]) / 2.0
    signals["p_market_side_at_execution"] = np.where(
        signals["side"].eq("up"),
        up_mid_execution / (up_mid_execution + down_mid_execution),
        down_mid_execution / (up_mid_execution + down_mid_execution),
    )
    signals["market_probability_change_after_start"] = (
        signals["p_market_side_at_execution"] - signals["p_market_side"]
    )
    signals["official_outcome_selected_side"] = np.where(
        signals["side"].eq("up"),
        signals["target_polymarket_up"],
        1 - signals["target_polymarket_up"],
    ).astype(int)
    signals["market_price_bin"] = pd.cut(
        signals["p_market_side"], PRICE_BINS, labels=PRICE_LABELS, right=False
    )
    signals["model_market_gap_bin"] = pd.cut(
        signals["model_market_probability_gap"],
        GAP_BINS,
        labels=GAP_LABELS,
        right=False,
    )

    if len(signals) != independent_summary["trades"]:
        raise AssertionError("Signal table row count differs from the independent replay")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = OUTPUT_DIR / "all_independent_btc_platt_entry_signals_1s.parquet"
    signals.to_parquet(artifact, index=False)
    price_scores = _score_rows(signals, ["fold", "side", "market_price_bin"])
    gap_scores = _score_rows(signals, ["fold", "side", "model_market_gap_bin"])
    price_scores["group_type"] = "market_probability"
    gap_scores["group_type"] = "model_market_gap"
    bins = pd.concat([price_scores, gap_scores], ignore_index=True)
    bins = bins[[
        "group_type", "fold", "side", "market_price_bin", "model_market_gap_bin",
        "n", "mean_model_probability", "mean_market_probability", "actual_win_rate",
        "mean_execution_price", "mean_break_even_probability", "mean_expected_net_pnl_usd",
        "pnl_after_cost_usd",
    ]]
    by_replay_status = _score_rows(signals, ["fold", "side", "portfolio_rejection_reason"])
    summary = {
        "evaluation_source": str((RUN_DIR / "shared_market_evaluation.parquet").relative_to(ROOT)),
        "quote_source": str(QUOTE_PATH.relative_to(ROOT)),
        "model_variant": "new_btc_platt",
        "execution_delay_seconds": EXECUTION_DELAY_SECONDS,
        "stake_usd": STAKE_USD,
        "portfolio_bankroll_usd": BANKROLL_USD,
        "minimum_expected_pnl_usd": MIN_EXPECTED_PNL_USD,
        "independent_signal_replay": independent_summary,
        "capital_limited_portfolio_replay": portfolio_summary,
        "signal_rows": len(signals),
        "portfolio_executed_rows": len(portfolio_ids),
        "portfolio_cash_rejected_rows": cash_rejected,
        "time_ranges_by_fold": signals.groupby("fold", observed=True).agg(
            start=("market_start_utc", "min"), end=("market_start_utc", "max"), n=("condition_id", "size")
        ).reset_index().to_dict(orient="records"),
        "overall": _score_rows(signals, ["side"]).to_dict(orient="records"),
        "portfolio_status_by_fold_and_side": by_replay_status.to_dict(orient="records"),
        "predeclared_bins": bins.to_dict(orient="records"),
        "quote_timing": {
            "first_quote_delay_seconds": signals["first_quote_delay_after_prediction_seconds"].describe().to_dict(),
            "execution_quote_delay_after_prediction_seconds": signals["execution_quote_delay_after_prediction_seconds"].describe().to_dict(),
            "execution_quote_delay_after_market_start_seconds": signals["execution_quote_delay_after_market_start_seconds"].describe().to_dict(),
            "execution_quote_selection_delay_ms": signals["quote_delay_ms"].describe().to_dict(),
            "exchange_quote_age_known_count": int(signals["book_age_known"].fillna(False).sum()),
            "stale_execution_quote_count": int(signals["stale_observation"].fillna(False).sum()),
            "market_probability_change_after_start": signals["market_probability_change_after_start"].describe().to_dict(),
        },
        "artifact": str(artifact.relative_to(ROOT)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_summary = OUTPUT_DIR / "entry_selection_summary_20261003.json"
    output_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_native) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "signal_rows": len(signals),
        "portfolio_executed_rows": len(portfolio_ids),
        "cash_rejected_rows": cash_rejected,
        "independent_replay": independent_summary,
        "portfolio_replay": portfolio_summary,
        "quote_timing": summary["quote_timing"],
        "artifacts": [str(artifact), str(output_summary)],
        "elapsed_seconds": summary["elapsed_seconds"],
    }, indent=2, ensure_ascii=False, default=_native))


if __name__ == "__main__":
    main()
