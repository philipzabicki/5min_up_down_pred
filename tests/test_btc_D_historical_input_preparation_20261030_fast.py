import copy
import hashlib
import json
import unittest

import pandas as pd
import pyarrow as pa

from features import btc_D_historical_input_preparation_20261030 as e0048
from features import btc_D_historical_input_preparation_20261030_fast as fast
import run_btc_preopen_economic_replay as replay


def _event(received, source, market, asset, event_type, *, bids="[]", asks="[]", price=None, size=None, side=None, best_bid=None, best_ask=None, fee_rate_bps=None):
    return {
        "timestamp_received": received,
        "timestamp": source,
        "market": market,
        "event_type": event_type,
        "asset_id": asset,
        "bids": bids,
        "asks": asks,
        "price": price,
        "size": size,
        "side": side,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "fee_rate_bps": fee_rate_bps,
    }


def _table(rows):
    return pa.Table.from_pylist(rows).select(list(e0048.EVENT_COLUMNS))


class E0048FastReplayParityTests(unittest.TestCase):
    def test_vectorized_token_filter_matches_frozen_filter(self):
        deadline = pd.Timestamp("2026-09-09T05:04:01Z")
        t60 = deadline - pd.Timedelta(seconds=1)
        rows = [
            _event(t60, t60, b"a", "up-a", "book"),
            _event(deadline, deadline, b"a", "up-a", "book"),
            _event(deadline + pd.Timedelta(1, unit="us"), deadline, b"a", "up-a", "book"),
            _event(t60, t60, b"a", "not-a-token", "book"),
            _event(t60, t60, b"outside", "up-a", "book"),
            _event(None, t60, b"a", "up-a", "book"),
            _event(t60, None, b"a", "up-a", "book"),
        ]
        table = _table(rows)
        args = (table, 10, {"a"}, {"a": int(deadline.value)}, {"a": {"up-a", "down-a"}})
        expected, expected_unknown = e0048._as_event_frame(*args)
        observed, observed_unknown = fast.as_event_frame(*args)
        pd.testing.assert_frame_equal(expected, observed, check_exact=True, check_dtype=True)
        self.assertEqual(expected_unknown, {"a"})
        self.assertEqual(observed_unknown, expected_unknown)
        self.assertEqual(expected["_row_order"].tolist(), [10, 11])
        self.assertEqual(expected["_received_ns"].tolist(), [t60.value, deadline.value])

    def test_ordered_replay_preserves_split_receive_groups_conflicts_and_cutoffs(self):
        start_a = pd.Timestamp("2026-09-09T05:05:00Z")
        start_b = pd.Timestamp("2026-09-09T05:05:02Z")
        t60_a = start_a - pd.Timedelta(seconds=60)
        t59_a = start_a - pd.Timedelta(seconds=59)
        t60_b = start_b - pd.Timedelta(seconds=60)
        t59_b = start_b - pd.Timedelta(seconds=59)
        raw = [
            _event(t60_a, t60_a, "a", "up-a", "book", bids="[[0.40,10]]", asks="[[0.55,5]]"),
            _event(t60_a, t60_a, "b", "up-b", "best_bid_ask", best_bid=0.20, best_ask=0.80),
            _event(t60_a, t60_a, "a", "up-a", "book", bids="[[0.41,10]]", asks="[[0.56,5]]"),
            _event(t59_a, t59_a, "a", "up-a", "price_change", price=0.60, size=1.0, side="SELL"),
            _event(t59_a, t59_a, "b", "down-b", "last_trade_price", fee_rate_bps=1.25),
            _event(t59_a, t59_a, "a", "up-a", "price_change", price=0.60, size=2.0, side="SELL"),
            _event(t60_b, t60_b, "b", "down-b", "book", bids="[[0.33,7]]", asks="[[0.67,8]]"),
            _event(t59_b, t59_b, "b", "down-b", "last_trade_price", fee_rate_bps=2.5),
            _event(t59_a + pd.Timedelta(1, unit="us"), t59_a, "a", "up-a", "price_change", price=0.61, size=3.0, side="SELL"),
            _event(t60_a, t60_a, "a", "wrong-token", "book"),
            _event(t60_a, t60_a, "outside", "up-a", "book"),
            _event(None, t60_a, "a", "up-a", "book"),
        ]
        source = _table(raw)
        chunks = (source.slice(0, 1), source.slice(1, 3), source.slice(4, 3), source.slice(7))
        active = {"a", "b"}
        tokens = {"a": {"up-a", "down-a"}, "b": {"up-b", "down-b"}}
        deadlines = {"a": int(t59_a.value), "b": int(t59_b.value)}
        old_frames, fast_frames = [], []
        old_unknown, fast_unknown = set(), set()
        row_offset = 0
        for chunk in chunks:
            old_frame, old_missing = e0048._as_event_frame(chunk, row_offset, active, deadlines, tokens)
            fast_frame, fast_missing = fast.as_event_frame(chunk, row_offset, active, deadlines, tokens)
            pd.testing.assert_frame_equal(old_frame, fast_frame, check_exact=True, check_dtype=True)
            old_frames.append(old_frame)
            fast_frames.append(fast_frame)
            old_unknown.update(old_missing)
            fast_unknown.update(fast_missing)
            row_offset += len(chunk)
        self.assertEqual(old_unknown, {"a"})
        self.assertEqual(fast_unknown, old_unknown)
        old_events = pd.concat(old_frames, ignore_index=True)
        fast_events = pd.concat(fast_frames, ignore_index=True)
        pd.testing.assert_frame_equal(old_events, fast_events, check_exact=True, check_dtype=True)
        self.assertEqual(len(old_events), 8)

        markets = {
            "a": {"condition_id": "a", "market_start_utc": start_a, "up_token_id": "up-a", "down_token_id": "down-a"},
            "b": {"condition_id": "b", "market_start_utc": start_b, "up_token_id": "up-b", "down_token_id": "down-b"},
        }
        t60 = [(int(t60_a.value), "a"), (int(t60_b.value), "b")]
        t59 = [(int(t59_a.value), "a"), (int(t59_b.value), "b")]
        processed_hours = {t60_a.floor("h")}

        def make_capture(states, snapshots60, snapshots59):
            i60 = i59 = 0
            calls = []

            def capture(cutoff_ns, inclusive):
                nonlocal i60, i59
                calls.append((int(cutoff_ns), bool(inclusive)))
                while i60 < len(t60) and (t60[i60][0] < cutoff_ns or (inclusive and t60[i60][0] == cutoff_ns)):
                    deadline, condition_id = t60[i60]
                    state = states.get(condition_id, replay._new_state())
                    snapshots60[condition_id] = copy.deepcopy(state)
                    i60 += 1
                while i59 < len(t59) and (t59[i59][0] < cutoff_ns or (inclusive and t59[i59][0] == cutoff_ns)):
                    deadline, condition_id = t59[i59]
                    state = states.get(condition_id, replay._new_state())
                    snapshots59[condition_id] = copy.deepcopy(state)
                    states.pop(condition_id, None)
                    i59 += 1

            return capture, calls, lambda: (i60, i59)

        def replay_old(events):
            states = {}
            for condition_id in old_unknown & active:
                states.setdefault(condition_id, replay._new_state())["unknown_event_timing"] = True
            snapshots60, snapshots59 = {}, {}
            capture, calls, indices = make_capture(states, snapshots60, snapshots59)
            events.sort_values(["_received_ns", "_row_order"], kind="stable", inplace=True)
            digest = hashlib.sha256()
            replayed = groups = multi = max_group = 0
            for received_ns, receive_frame in events.groupby("_received_ns", sort=False):
                received_ns = int(received_ns)
                capture(received_ns, False)
                market_count = 0
                for condition_id, group in receive_frame.groupby("market", sort=False):
                    rows = list(group.itertuples(index=False))
                    digest.update(f"{received_ns}|{condition_id}|".encode())
                    for order in group["_row_order"].to_numpy(copy=False):
                        digest.update(int(order).to_bytes(8, "big", signed=False))
                    state = states.setdefault(str(condition_id), replay._new_state())
                    e0048._apply_native_receive_group(state, rows, received_ns)
                    replayed += len(rows)
                    groups += 1
                    market_count += 1
                    max_group = max(max_group, len(rows))
                multi += int(market_count > 1)
                capture(received_ns, True)
            capture(int((t59_b + pd.Timedelta(1, unit="us")).value), False)
            self_indices = indices()
            return states, snapshots60, snapshots59, calls, digest.hexdigest(), replayed, groups, multi, max_group, self_indices

        old = replay_old(old_events.copy(deep=True))
        states = {}
        for condition_id in fast_unknown & active:
            states.setdefault(condition_id, replay._new_state())["unknown_event_timing"] = True
        snapshots60, snapshots59 = {}, {}
        capture, calls, indices = make_capture(states, snapshots60, snapshots59)
        fast_stats = fast.replay_ordered_events(fast_events.copy(deep=True), states, capture, replay)
        capture(int((t59_b + pd.Timedelta(1, unit="us")).value), False)
        self.assertEqual(indices(), (len(t60), len(t59)))

        self.assertEqual(old[0], states)
        self.assertEqual(old[1], snapshots60)
        self.assertEqual(old[2], snapshots59)
        self.assertEqual(old[3], calls)
        self.assertEqual(old[4], fast_stats["group_order_sha256"])
        self.assertEqual(old[5], fast_stats["replayed_rows"])
        self.assertEqual(old[6], fast_stats["market_group_count"])
        self.assertEqual(old[7], fast_stats["multi_market_receive_group_count"])
        self.assertEqual(old[8], fast_stats["max_market_group_rows"])
        self.assertEqual(old[9], indices())
        self.assertEqual(old[5], 8)
        self.assertEqual(old[7], 2)

        self.assertEqual(snapshots60["a"]["event_count"], 2)
        self.assertTrue(snapshots60["a"]["unknown_event_timing"])
        self.assertTrue(snapshots60["a"]["tokens"]["up-a"]["bid_order_ambiguous"])
        self.assertTrue(snapshots60["a"]["tokens"]["up-a"]["ask_order_ambiguous"])
        self.assertEqual(snapshots59["a"]["event_count"], 4)
        self.assertEqual(snapshots59["a"]["equal_source_timestamp_price_changes"], 1)
        self.assertEqual(snapshots59["a"]["tokens"]["up-a"]["asks"][0.60], 2.0)
        self.assertEqual(snapshots60["b"]["tokens"]["down-b"]["bids"][0.33], 7.0)
        self.assertEqual(snapshots59["b"]["fee_rate_bps"], 2.5)


if __name__ == "__main__":
    unittest.main()