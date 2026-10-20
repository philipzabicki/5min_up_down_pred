"""Pure helpers for the BTC market-dynamics action-value study."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


DEVELOPMENT_CUTOFF_UTC = pd.Timestamp("2026-07-01T00:00:00Z")
ACTION_SIZES_USD = (5.0, 10.0, 25.0, 50.0, 100.0)
HORIZONS_SECONDS = (5, 15, 60)
MODEL_FEATURES_STATIC = (
    "raw_logit",
    "market_logit",
    "raw_minus_market_logit",
    "spread_up",
    "spread_down",
    "mid_sum_minus_one",
    "log1p_ask_notional_up",
    "log1p_ask_notional_down",
    "slippage_5_up",
    "slippage_5_down",
)
MODEL_FEATURES_DYNAMIC = tuple(
    f"{name}_{h}s"
    for h in HORIZONS_SECONDS
    for name in (
        "p_market_change",
        "spread_up_change",
        "spread_down_change",
        "log1p_ask_notional_up_change",
        "log1p_ask_notional_down_change",
        "ask_depletion_relative_up",
        "ask_depletion_relative_down",
    )
)


def development_mask(values: Iterable, cutoff=DEVELOPMENT_CUTOFF_UTC) -> np.ndarray:
    """Return a strict market-start cutoff mask, coercing timestamps to UTC."""
    starts = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    return starts.lt(pd.Timestamp(cutoff)).to_numpy(dtype=bool)


def assert_development_starts(
    values: Iterable,
    *,
    cutoff=DEVELOPMENT_CUTOFF_UTC,
    context: str = "table",
) -> None:
    starts = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if starts.isna().any():
        raise ValueError(f"{context} contains a missing/invalid market start")
    bad = starts.ge(pd.Timestamp(cutoff))
    if bad.any():
        first = starts.loc[bad].min().isoformat()
        raise ValueError(f"{context} contains a non-development market start: {first}")


def clipped_logit(probability: float) -> float:
    p = float(np.clip(float(probability), 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


def ask_band_notional(levels: Mapping[float, float], best_ask: float) -> float:
    low = float(best_ask)
    high = min(1.0, low + 0.02)
    return float(sum(float(price) * float(size) for price, size in levels.items()
                     if low <= float(price) <= high and float(size) > 0.0))


def gross_vwap(levels: Mapping[float, float], requested_usd: float) -> float | None:
    remaining = float(requested_usd)
    shares = 0.0
    for price in sorted(levels):
        price = float(price)
        size = float(levels[price])
        if not (0.0 < price < 1.0) or size <= 0.0:
            continue
        take = min(remaining / price, size)
        shares += take
        remaining -= take * price
        if remaining <= 1e-9:
            return float(requested_usd) / shares if shares > 0.0 else None
    return None


def level_depletion_notional(
    previous_size: float | None,
    new_size: float,
    price: float,
    *,
    event_type: str = "price_change",
    side: str = "SELL",
    native: bool = True,
    in_fixed_band: bool,
) -> float:
    """Notional removed by an observed native level update; snapshots never call this."""
    if event_type != "price_change" or side.upper() != "SELL" or not native or not in_fixed_band:
        return 0.0
    previous = 0.0 if previous_size is None else max(0.0, float(previous_size))
    current = max(0.0, float(new_size))
    return float(price) * max(previous - current, 0.0)


def in_open_closed_window(received_ns: int, start_ns: int, end_ns: int) -> bool:
    return int(start_ns) < int(received_ns) <= int(end_ns)


def book_initialization_precedes(first_book_receive_ns: int | None, oldest_endpoint_ns: int) -> bool:
    return first_book_receive_ns is not None and int(first_book_receive_ns) < int(oldest_endpoint_ns)


def causal_event_mask(received_ns: Iterable, source_ns: Iterable, cutoff_ns: int) -> np.ndarray:
    received = np.asarray(list(received_ns), dtype=np.int64)
    source = np.asarray(list(source_ns), dtype=np.int64)
    if received.shape != source.shape:
        raise ValueError("receive and source timestamp arrays must have equal length")
    return (received <= int(cutoff_ns)) & (source <= int(cutoff_ns))


def action_return(
    *,
    side: str,
    outcome_up: float | None,
    net_shares: float | None,
    cash_debit_usd: float | None,
    fill_status: str | None,
    label_known: bool,
) -> tuple[float | None, str]:
    """Return r and status without collapsing unknown execution into a no-fill."""
    if not label_known or outcome_up not in (0.0, 1.0):
        return None, "unsettled_or_label_unknown"
    if fill_status in (None, "missing", "unknown_execution") or net_shares is None or cash_debit_usd is None:
        return None, "execution_unknown"
    if fill_status == "no_fill":
        return 0.0, "observed_no_fill"
    if fill_status not in {"full", "partial_fill"}:
        return None, "execution_unknown"
    won = (side == "up" and float(outcome_up) == 1.0) or (
        side == "down" and float(outcome_up) == 0.0
    )
    payout = float(net_shares) if won else 0.0
    debit = float(cash_debit_usd)
    if debit < 0.0 or payout < 0.0:
        raise ValueError("payout and cash debit must be nonnegative")
    return payout - debit, "settled"


def action_return_fraction(net_pnl_usd: float, requested_usd: float) -> float:
    requested = float(requested_usd)
    if requested <= 0.0:
        raise ValueError("requested_usd must be positive")
    return float(net_pnl_usd) / requested


def label_released_before(release_at, first_decision_at) -> bool:
    release = pd.Timestamp(release_at)
    decision = pd.Timestamp(first_decision_at)
    if release.tzinfo is None or decision.tzinfo is None:
        raise ValueError("release and decision timestamps must be timezone-aware")
    return bool(release.tz_convert("UTC") < decision.tz_convert("UTC"))


def market_action_weights(market_ids: Iterable) -> np.ndarray:
    ids = pd.Series(list(market_ids), dtype="string")
    counts = ids.groupby(ids, dropna=False).transform("size").to_numpy(dtype=np.float64)
    if len(counts) and (counts <= 0.0).any():
        raise ValueError("every action row must map to a market")
    return 1.0 / counts if len(counts) else np.empty(0, dtype=np.float64)


def assert_single_fold_per_market(market_ids: Iterable, fold_ids: Iterable) -> None:
    frame = pd.DataFrame({"market": list(market_ids), "fold": list(fold_ids)})
    if frame.market.isna().any() or frame.fold.isna().any():
        raise ValueError("market and fold identifiers must be present")
    if frame.groupby("market", dropna=False).fold.nunique().gt(1).any():
        raise ValueError("all action rows for a market must stay in one fold")


def choose_action(
    candidates: Iterable[Mapping],
    *,
    minimum_return: float = 0.01,
) -> Mapping | None:
    """Choose maximum net return, tie UP; qualification is applied after selection."""
    actions = [item for item in candidates if item.get("predicted_return") is not None]
    if not actions:
        return None
    chosen = max(actions, key=lambda item: (float(item["predicted_return"]), item["side"] == "up"))
    return chosen if float(chosen["predicted_return"]) >= float(minimum_return) else None


def action_size_outside_training_range(requested_usd: float) -> bool:
    return not (ACTION_SIZES_USD[0] <= float(requested_usd) <= ACTION_SIZES_USD[-1])


def action_descriptor(
    *,
    side: str,
    requested_usd: float,
    t60_vwap: float,
    t60_best_ask: float,
    t60_cash_debit_usd: float,
    t60_net_shares: float,
    t60_depth_known: bool,
) -> dict[str, float]:
    requested = float(requested_usd)
    if requested <= 0.0:
        raise ValueError("requested_usd must be positive")
    vwap = float(t60_vwap)
    return {
        "action_is_up": float(side == "up"),
        "log_requested_usd": math.log(requested),
        "t60_vwap": vwap,
        "t60_slippage": vwap - float(t60_best_ask),
        "t60_predicted_cash_debit_usd": float(t60_cash_debit_usd),
        "t60_predicted_net_shares": float(t60_net_shares),
        "t60_depth_known": float(bool(t60_depth_known)),
    }


def expected_net_pnl(
    *, probability_up: float, side: str, net_shares: float, cash_debit_usd: float,
) -> float:
    p_up = float(probability_up)
    p_side = p_up if side == "up" else 1.0 - p_up
    return p_side * float(net_shares) - float(cash_debit_usd)


def daily_status_series(
    start_day,
    end_day_exclusive,
    daily_values: Mapping,
    *,
    unknown_days: Iterable = (),
) -> pd.DataFrame:
    """Preserve empty calendar days as zero and unknown days as NaN with status."""
    days = pd.date_range(start_day, end_day_exclusive, freq="D", inclusive="left", tz="UTC")
    unknown = {pd.Timestamp(day).floor("D") for day in unknown_days}
    rows = []
    for day in days:
        key = day.date().isoformat()
        if day in unknown:
            rows.append({"day_utc": day, "value": np.nan, "status": "execution_unknown"})
        else:
            rows.append({"day_utc": day, "value": float(daily_values.get(key, 0.0)),
                         "status": "observed" if key in daily_values else "no_trade_or_no_market"})
    return pd.DataFrame(rows)


def checkpoint_matches(saved: Mapping | None, identity: Mapping) -> bool:
    return bool(saved and saved.get("identity") == dict(identity)
                and saved.get("status") == "completed")
