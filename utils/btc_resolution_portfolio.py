"""Self-financing accounting for BTC positions intentionally held to resolution."""

from __future__ import annotations


class ResolutionPortfolio:
    """Track buys, locked cost basis, and one-time official-resolution payouts."""

    def __init__(self, initial_cash_usd: float) -> None:
        if initial_cash_usd <= 0:
            raise ValueError("initial cash must be positive")
        self.initial_cash_usd = float(initial_cash_usd)
        self.cash_usd = float(initial_cash_usd)
        self.locked_cost_basis_usd = 0.0
        self.open_positions: dict[str, dict] = {}
        self.used_position_ids: set[str] = set()
        self.settled_position_ids: set[str] = set()
        self.events: list[dict] = []
        self.accepted_trades = 0
        self.rejected_insufficient_cash = 0
        self.settled_trades = 0
        self.gross_turnover_usd = 0.0
        self.cash_fees_usd = 0.0
        self.share_fees = 0.0
        self.fee_equivalent_usd = 0.0
        self.payouts_usd = 0.0
        self.peak_locked_cost_basis_usd = 0.0
        self._peak_equity_usd = self.initial_cash_usd
        self.max_drawdown_fraction = 0.0

    @property
    def equity_at_cost_basis_usd(self) -> float:
        return self.cash_usd + self.locked_cost_basis_usd

    def _record(self, event: dict) -> None:
        equity = self.equity_at_cost_basis_usd
        self._peak_equity_usd = max(self._peak_equity_usd, equity)
        if self._peak_equity_usd > 0.0:
            self.max_drawdown_fraction = max(
                self.max_drawdown_fraction,
                (self._peak_equity_usd - equity) / self._peak_equity_usd,
            )
        self.events.append({
            **event,
            "free_cash_usd": self.cash_usd,
            "locked_cost_basis_usd": self.locked_cost_basis_usd,
            "equity_at_cost_basis_usd": equity,
            "drawdown_fraction": self.max_drawdown_fraction,
        })

    def buy(
        self,
        *,
        position_id: str,
        side: str,
        gross_usd: float,
        cash_debit_usd: float,
        net_shares: float,
        cash_fee_usd: float,
        share_fee: float,
        fee_equivalent_usd: float = 0.0,
        official_outcome_up: int,
        market_end_utc,
        outcome_known_at_utc,
        payout_available_at_utc,
        entry_time_utc,
    ) -> bool:
        if position_id in self.used_position_ids:
            raise ValueError(f"duplicate position id: {position_id}")
        self.used_position_ids.add(position_id)
        if side not in {"up", "down"}:
            raise ValueError("side must be 'up' or 'down'")
        if official_outcome_up not in {0, 1}:
            raise ValueError("official outcome must be binary")
        gross_usd = float(gross_usd)
        cash_debit_usd = float(cash_debit_usd)
        net_shares = float(net_shares)
        cash_fee_usd = float(cash_fee_usd)
        share_fee = float(share_fee)
        fee_equivalent_usd = float(fee_equivalent_usd)
        if (
            gross_usd <= 0.0
            or net_shares < 0.0
            or cash_fee_usd < 0.0
            or share_fee < 0.0
            or fee_equivalent_usd < 0.0
        ):
            raise ValueError("invalid purchase amounts")
        if cash_debit_usd + 1e-9 < gross_usd:
            raise ValueError("cash debit cannot be below gross purchase notional")
        if payout_available_at_utc < max(market_end_utc, outcome_known_at_utc):
            raise ValueError("payout cannot be available before market end and official outcome")
        if cash_debit_usd > self.cash_usd + 1e-9:
            self.rejected_insufficient_cash += 1
            self._record({
                "event": "buy_rejected_insufficient_cash",
                "position_id": position_id,
                "event_time_utc": entry_time_utc,
                "side": side,
                "gross_usd": gross_usd,
                "cash_debit_usd": cash_debit_usd,
                "payout_usd": 0.0,
            })
            return False

        self.cash_usd -= cash_debit_usd
        self.locked_cost_basis_usd += gross_usd
        self.accepted_trades += 1
        self.gross_turnover_usd += gross_usd
        self.cash_fees_usd += cash_fee_usd
        self.share_fees += share_fee
        self.fee_equivalent_usd += fee_equivalent_usd
        self.peak_locked_cost_basis_usd = max(
            self.peak_locked_cost_basis_usd, self.locked_cost_basis_usd
        )
        self.open_positions[position_id] = {
            "position_id": position_id,
            "side": side,
            "gross_usd": gross_usd,
            "net_shares": net_shares,
            "official_outcome_up": official_outcome_up,
            "market_end_utc": market_end_utc,
            "outcome_known_at_utc": outcome_known_at_utc,
            "payout_available_at_utc": payout_available_at_utc,
        }
        self._record({
            "event": "buy_filled",
            "position_id": position_id,
            "event_time_utc": entry_time_utc,
            "side": side,
            "gross_usd": gross_usd,
            "cash_debit_usd": cash_debit_usd,
            "cash_fee_usd": cash_fee_usd,
            "share_fee": share_fee,
            "fee_equivalent_usd": fee_equivalent_usd,
            "net_shares": net_shares,
            "payout_usd": 0.0,
        })
        return True

    def release_payouts_available_by(self, timestamp_utc) -> int:
        ready = sorted(
            (
                position
                for position in self.open_positions.values()
                if position["payout_available_at_utc"] <= timestamp_utc
            ),
            key=lambda position: (position["payout_available_at_utc"], position["position_id"]),
        )
        for position in ready:
            won = (
                position["official_outcome_up"] == 1
                if position["side"] == "up"
                else position["official_outcome_up"] == 0
            )
            payout = position["net_shares"] if won else 0.0
            self.cash_usd += payout
            self.locked_cost_basis_usd -= position["gross_usd"]
            self.payouts_usd += payout
            del self.open_positions[position["position_id"]]
            self.settled_position_ids.add(position["position_id"])
            self.settled_trades += 1
            self._record({
                "event": "official_resolution_payout",
                "position_id": position["position_id"],
                "event_time_utc": position["payout_available_at_utc"],
                "side": position["side"],
                "gross_usd": position["gross_usd"],
                "cash_debit_usd": 0.0,
                "payout_usd": payout,
                "winning_token": won,
                "market_end_utc": position["market_end_utc"],
                "outcome_known_at_utc": position["outcome_known_at_utc"],
                "payout_available_at_utc": position["payout_available_at_utc"],
            })
        return len(ready)

    def summary(self) -> dict:
        locked_positions = list(self.open_positions.values())
        open_locked = sum(position["gross_usd"] for position in locked_positions)
        if self.cash_usd < -1e-9:
            raise RuntimeError("portfolio cash is negative")
        if abs(self.locked_cost_basis_usd - open_locked) > 1e-8:
            raise RuntimeError("locked cost basis does not match open positions")
        if self.accepted_trades != self.settled_trades + len(locked_positions):
            raise RuntimeError("accepted positions are missing or settled more than once")
        if self.settled_position_ids.intersection(self.open_positions):
            raise RuntimeError("a position is both open and settled")
        return {
            "initial_cash_usd": self.initial_cash_usd,
            "accepted_trades": self.accepted_trades,
            "rejected_insufficient_cash": self.rejected_insufficient_cash,
            "settled_trades": self.settled_trades,
            "gross_turnover_usd": self.gross_turnover_usd,
            "cash_fees_usd": self.cash_fees_usd,
            "share_fees": self.share_fees,
            "total_fee_equivalent_usd": self.fee_equivalent_usd,
            "payouts_usd": self.payouts_usd,
            "free_cash_usd": self.cash_usd,
            "locked_cost_basis_usd": self.locked_cost_basis_usd,
            "peak_locked_cost_basis_usd": self.peak_locked_cost_basis_usd,
            "ending_equity_at_cost_basis_usd": self.equity_at_cost_basis_usd,
            "net_pnl_usd": self.equity_at_cost_basis_usd - self.initial_cash_usd,
            "max_drawdown_fraction": self.max_drawdown_fraction,
            "unsettled_positions": len(locked_positions),
            "unsettled_locked_cost_basis_usd": sum(p["gross_usd"] for p in locked_positions),
            "unsettled_position_ids": [p["position_id"] for p in locked_positions],
        }
