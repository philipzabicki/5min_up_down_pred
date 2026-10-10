import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from utils import polymarket_btc_exit_quotes as exit_quotes
from utils import polymarket_btc5m_fee_rules as fee_rules
from run_btc_historical_policy_study import _simulate_exit_portfolio


class PolymarketBtcExitQuoteTests(unittest.TestCase):
    def _market(self):
        start = pd.Timestamp("2026-04-15T17:05:00Z")
        return {
            "condition_id": "0xmarket",
            "entry_ns": int((start - pd.Timedelta(seconds=59)).value),
            "market_start_ns": int(start.value),
            "up_token_id": "up-token",
            "down_token_id": "down-token",
            "snapshot": {
                "_up_levels": ((0.51, 100.0),),
                "_down_levels": ((0.49, 100.0),),
            },
        }

    def _price_change(self, received_ns, price, size):
        return {
            "market": "0xmarket",
            "timestamp_received": pd.Timestamp(received_ns, unit="ns", tz="UTC"),
            "timestamp": pd.Timestamp(received_ns, unit="ns", tz="UTC"),
            "event_type": "price_change",
            "asset_id": "up-token",
            "bids": None,
            "asks": None,
            "price": price,
            "size": size,
            "side": "BUY",
            "best_bid": price,
            "best_ask": 0.51,
            "fee_rate_bps": None,
        }

    def test_event_path_does_not_use_a_later_receive_update(self):
        market = self._market()
        start_ns = market["market_start_ns"]
        events = pd.DataFrame([
            self._price_change(start_ns + 4_000_000_000, 0.49, 10.0),
            self._price_change(start_ns + 6_000_000_000, 0.49, 0.0),
            self._price_change(start_ns + 6_000_000_000, 0.48, 12.0),
            self._price_change(start_ns + 6_500_000_000, 0.47, 14.0),
        ])
        result = exit_quotes.quote_path_from_events(market, events, source_complete=True)
        sample = result["quotes"]["5"]
        self.assertEqual(sample["decision"]["up"]["levels"], [[0.49, 10.0]])
        self.assertEqual(sample["arrival"]["up"]["levels"], [[0.48, 12.0]])
        self.assertEqual(sample["arrival"]["up"]["quote_received_ns"], start_ns + 6_000_000_000)
        self.assertIsNone(result["quotes"]["10"]["decision"]["up"])

    def test_actual_cache_stage_resumes_and_extracts_only_new_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "event_parts"
            archive.mkdir()
            tick_path = root / "ticks.parquet"
            pd.DataFrame({
                "condition_id": pd.Series(dtype="string"),
                "t": pd.Series(dtype="int64"),
                "bu": pd.Series(dtype="float64"), "bd": pd.Series(dtype="float64"),
                "su": pd.Series(dtype="float64"), "sd": pd.Series(dtype="float64"),
            }).to_parquet(tick_path)
            manifest_path = root / "archive_manifest.json"
            manifest_path.write_text(json.dumps({"source_identity_sha256": "test-source"}), encoding="utf-8")
            index_path = root / "market_index.parquet"
            pd.DataFrame({
                "condition_id": ["0xmarket", "0xmarket2"],
                "archive_history_complete": [True, True],
            }).to_parquet(index_path)
            snapshot_path = root / "t59.parquet"
            snapshot_path.write_bytes(b"frozen-test-snapshot")
            cache_dir = root / "cache"
            first_market = self._market()
            second_market = {**first_market, "condition_id": "0xmarket2"}
            calls = []

            def fake_pmxt(day_markets, *, archive_dir, index_by_id):
                calls.append([row["condition_id"] for row in day_markets])
                return [{
                    "condition_id": row["condition_id"],
                    "source": "pmxt_event_replay",
                    "source_complete": True,
                    "missing_hours": [],
                    "quotes": {"5": {"decision": {"up": None}, "arrival": {"up": None}}},
                } for row in day_markets]

            args = {
                "cache_dir": cache_dir,
                "archive_dir": archive,
                "kacho_ticks_path": tick_path,
                "manifest_path": manifest_path,
                "index_path": index_path,
                "snapshot_path": snapshot_path,
                "progress": False,
            }
            with patch.object(exit_quotes, "_read_kacho_paths", return_value={}), patch.object(
                exit_quotes, "_pmxt_paths_for_day", side_effect=fake_pmxt,
            ):
                first, first_report = exit_quotes.extract_exit_paths([first_market], **args)
                second, second_report = exit_quotes.extract_exit_paths([first_market], **args)
                expanded, expanded_report = exit_quotes.extract_exit_paths(
                    [first_market, second_market], **args,
                )

            self.assertEqual(len(calls), 2)
            self.assertEqual(first_report["extracted_count"], 1)
            self.assertEqual(second_report["cache_hit_count"], 1)
            self.assertEqual(second_report["extracted_count"], 0)
            self.assertEqual(expanded_report["cache_hit_count"], 1)
            self.assertEqual(expanded_report["extracted_count"], 1)
            self.assertEqual(set(expanded), {"0xmarket", "0xmarket2"})
            self.assertEqual(second["0xmarket"]["quotes"]["5"]["decision"]["up"], None)

    def test_threshold_exit_uses_net_bid_proceeds_and_reconciles_cash(self):
        start = pd.Timestamp("2026-05-10T12:05:00Z")
        rule = fee_rules.resolve_market_fee_rule(None, start - pd.Timedelta(seconds=59))
        snapshot = {
            "up_quote_complemented": False, "up_quote_source_token_id": "up-token",
            "archive_entry_hour_available": True, "archive_gap_after_up_book_init": False,
            "_up_levels": ((0.5, 100.0),), "no_future_event_at_entry": True,
            "no_future_source_event_at_entry": True, "up_ask_order_ambiguous": False,
            "up_ask_age_seconds": 0.0, "up_bbo_ask_comparable": False,
            "down_quote_complemented": False, "down_quote_source_token_id": "down-token",
            "archive_gap_after_down_book_init": False, "_down_levels": ((0.9, 100.0),),
            "down_ask_order_ambiguous": False, "down_ask_age_seconds": 0.0,
            "down_bbo_ask_comparable": False,
        }
        market = {
            "condition_id": "0xmarket", "market_slug": "btc-updown-5m-test",
            "market_start_ns": int(start.value), "entry_ns": int((start-pd.Timedelta(seconds=59)).value),
            "market_start_utc": start, "resolved_ns": int((start+pd.Timedelta(seconds=310)).value),
            "resolved_at_utc": start+pd.Timedelta(seconds=310), "outcome": 0.0,
            "p_candidate_platt": 0.9, "up_token_id": "up-token", "down_token_id": "down-token",
            "snapshot": snapshot, "fee_rule": rule,
        }
        path = {
            "condition_id": "0xmarket", "source": "pmxt_event_replay", "source_complete": True,
            "quotes": {
                "5": {
                    "decision": {"up": {"levels": [[0.75, 20.0]], "age_seconds": 0.0}, "down": None},
                    "arrival": {"up": {"levels": [[0.74, 20.0]], "age_seconds": 0.0,
                                         "quote_received_ns": int((start+pd.Timedelta(seconds=6)).value)}, "down": None},
                },
            },
        }
        scenario = {
            "role": "test", "scenario_id": "fixed5", "policy": "fixed_5_usd",
            "cap_usd": None, "sizing_id": "fixed_5_usd", "minimum_net_return": 0.0,
        }
        summary, _, trades, missing = _simulate_exit_portfolio(
            [market], scenario=scenario,
            exit_policy={"id": "tp10_sl10", "kind": "threshold", "take_profit": 0.1, "stop_loss": -0.1},
            quote_paths={"0xmarket": path}, common_market_ids={"0xmarket"},
        )
        self.assertEqual(missing, set())
        self.assertEqual(summary["sold_count"], 1)
        self.assertEqual(trades[0]["actual_exit_type"], "sale")
        self.assertEqual(trades[0]["exit_trigger_offset_seconds"], 5)
        self.assertAlmostEqual(trades[0]["exit_net_proceeds_usd"], 7.26532)
        self.assertAlmostEqual(summary["ending_cash_after_all_releases_usd"], 102.09032)
        self.assertAlmostEqual(summary["fees_paid_estimated_usd"], 0.30968)

    def test_unfilled_fixed_close_holds_to_official_settlement(self):
        start = pd.Timestamp("2026-05-10T12:05:00Z")
        rule = fee_rules.resolve_market_fee_rule(None, start - pd.Timedelta(seconds=59))
        market = {
            **self._market(),
            "market_start_ns": int(start.value), "entry_ns": int((start-pd.Timedelta(seconds=59)).value),
            "market_start_utc": start, "resolved_ns": int((start+pd.Timedelta(seconds=310)).value),
            "resolved_at_utc": start+pd.Timedelta(seconds=310), "outcome": 1.0,
            "p_candidate_platt": 0.9, "fee_rule": rule,
        }
        market["snapshot"].update({
            "up_quote_source_token_id": "up-token", "down_quote_source_token_id": "down-token",
            "_up_levels": ((0.5, 100.0),), "_down_levels": ((0.9, 100.0),),
            "up_quote_complemented": False, "down_quote_complemented": False,
            "archive_entry_hour_available": True, "archive_gap_after_up_book_init": False,
            "archive_gap_after_down_book_init": False, "no_future_event_at_entry": True,
            "no_future_source_event_at_entry": True, "up_ask_order_ambiguous": False,
            "down_ask_order_ambiguous": False, "up_ask_age_seconds": 0.0,
            "down_ask_age_seconds": 0.0, "up_bbo_ask_comparable": False,
            "down_bbo_ask_comparable": False,
        })
        path = {
            "source": "pmxt_event_replay", "source_complete": True,
            "quotes": {"60": {"arrival": {"up": {
                "levels": [[0.5, 4.0]], "age_seconds": 0.0,
                "quote_received_ns": int((start+pd.Timedelta(seconds=61)).value),
            }, "down": None}}},
        }
        scenario = {
            "role": "test", "scenario_id": "fixed5", "policy": "fixed_5_usd",
            "cap_usd": None, "sizing_id": "fixed_5_usd", "minimum_net_return": 0.0,
        }
        summary, _, trades, _ = _simulate_exit_portfolio(
            [market], scenario=scenario,
            exit_policy={"id": "close_at_60s", "kind": "time", "offset_seconds": 60},
            quote_paths={"0xmarket": path}, common_market_ids={"0xmarket"},
        )
        self.assertEqual(trades[0]["exit_status"], "time_exit_unfilled_hold")
        self.assertEqual(trades[0]["actual_exit_type"], "resolution")
        self.assertAlmostEqual(summary["ending_cash_after_all_releases_usd"], 104.825)


if __name__ == "__main__":
    unittest.main()
