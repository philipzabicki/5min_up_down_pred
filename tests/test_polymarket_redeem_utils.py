import unittest

from utils.polymarket import (
    DEFAULT_POLYMARKET_CTF_COLLATERAL_ADAPTER_ADDRESS,
    DEFAULT_POLYMARKET_PUSD_ADDRESS,
    build_redeem_transactions,
    collect_redeem_candidates,
    encode_redeem_positions_call,
    polymarket_market_slug_matches_prefix,
    resolve_redeem_collateral_address,
    resolve_redeem_target_address,
    resolve_relayer_tx_type,
)

CONDITION_ID = "0x" + "12" * 32
ALT_CONDITION_ID = "0x" + "34" * 32
MARKET_PREFIX = "btc-updown-5m"


def _record(condition_id=CONDITION_ID, **overrides):
    rec = {
        "pm_mode": "live",
        "pm_condition_id": condition_id,
        "pm_selected_token_id": "123",
        "trade_side": "yes",
        "resolved_at": "2026-05-04T12:00:00+00:00",
        "actual_up": 1,
        "pm_settlement_status": "resolved_waiting_settlement",
        "pm_redeem_tx_id": "",
        "pm_redeem_tx_state": "",
    }
    rec.update(overrides)
    return rec


def _position(condition_id=CONDITION_ID, **overrides):
    pos = {
        "conditionId": condition_id,
        "asset": "123",
        "slug": f"{MARKET_PREFIX}-1770000000",
        "redeemable": True,
        "negativeRisk": False,
    }
    pos.update(overrides)
    return pos


class RedeemEncodingTests(unittest.TestCase):
    def test_encode_redeem_positions_call(self):
        calldata = encode_redeem_positions_call(CONDITION_ID)

        self.assertTrue(calldata.startswith("0x01b7037c"))
        self.assertIn(DEFAULT_POLYMARKET_PUSD_ADDRESS[2:].lower(), calldata)
        self.assertIn(CONDITION_ID[2:].lower(), calldata)
        self.assertTrue(calldata.endswith("".join([f"{1:064x}", f"{2:064x}"])))

    def test_reject_invalid_condition_id(self):
        with self.assertRaises(ValueError):
            encode_redeem_positions_call("0x1234")

    def test_collateral_address_override(self):
        override = "0x1111111111111111111111111111111111111111"

        self.assertEqual(
            resolve_redeem_collateral_address(
                {"POLY_REDEEM_COLLATERAL_TOKEN_ADDRESS": override}
            ),
            override,
        )

    def test_redeem_target_default_is_adapter(self):
        self.assertEqual(
            resolve_redeem_target_address({}),
            DEFAULT_POLYMARKET_CTF_COLLATERAL_ADAPTER_ADDRESS,
        )

    def test_relayer_tx_type_selection(self):
        self.assertEqual(resolve_relayer_tx_type({}, signature_type=2), "SAFE")
        self.assertEqual(resolve_relayer_tx_type({}, signature_type=3), "WALLET")
        self.assertEqual(
            resolve_relayer_tx_type({"POLY_RELAYER_TX_TYPE": "safe"}),
            "SAFE",
        )


class RedeemCandidateTests(unittest.TestCase):
    def test_market_slug_prefix_match_is_segment_safe(self):
        self.assertTrue(
            polymarket_market_slug_matches_prefix(
                f"{MARKET_PREFIX}-1770000000", MARKET_PREFIX
            )
        )
        self.assertFalse(
            polymarket_market_slug_matches_prefix(
                f"{MARKET_PREFIX}extra-1770000000", MARKET_PREFIX
            )
        )
        self.assertFalse(polymarket_market_slug_matches_prefix("", MARKET_PREFIX))
        self.assertFalse(polymarket_market_slug_matches_prefix(MARKET_PREFIX, ""))

    def test_skip_positions_outside_market_prefix(self):
        candidates, diagnostics = collect_redeem_candidates(
            [_position(slug="some-other-market-1770000000")],
            [_record()],
            market_slug_prefix=MARKET_PREFIX,
            require_redeemable=True,
        )

        self.assertEqual(candidates, [])
        self.assertEqual(diagnostics, [])

    def test_dedupe_condition_id(self):
        positions = [
            _position(asset="123"),
            _position(asset="123"),
            _position(condition_id=ALT_CONDITION_ID, asset="456"),
        ]
        records = [
            _record(),
            _record(condition_id=ALT_CONDITION_ID, pm_selected_token_id="456"),
        ]

        candidates, diagnostics = collect_redeem_candidates(
            positions,
            records,
            market_slug_prefix=MARKET_PREFIX,
            require_redeemable=True,
        )

        self.assertEqual([c["conditionId"] for c in candidates], [CONDITION_ID, ALT_CONDITION_ID])
        self.assertIn("duplicate_condition_id", {d["reason"] for d in diagnostics})

    def test_skip_pending_and_confirmed(self):
        positions = [
            _position(),
            _position(condition_id=ALT_CONDITION_ID, asset="456"),
        ]
        records = [
            _record(
                pm_redeem_tx_id="tx-pending",
                pm_redeem_tx_state="STATE_NEW",
                pm_settlement_status="redeem_submitted",
            ),
            _record(
                condition_id=ALT_CONDITION_ID,
                pm_selected_token_id="456",
                pm_redeem_tx_id="tx-confirmed",
                pm_redeem_tx_state="STATE_CONFIRMED",
                pm_settlement_status="redeem_confirmed_waiting_close_sync",
            ),
        ]

        candidates, diagnostics = collect_redeem_candidates(
            positions,
            records,
            market_slug_prefix=MARKET_PREFIX,
            require_redeemable=True,
        )

        self.assertEqual(candidates, [])
        reasons = {d["reason"] for d in diagnostics}
        self.assertIn("redeem_already_pending", reasons)
        self.assertIn("redeem_already_confirmed", reasons)

    def test_retry_failed_and_invalid(self):
        positions = [
            _position(),
            _position(condition_id=ALT_CONDITION_ID, asset="456"),
        ]
        records = [
            _record(
                pm_redeem_tx_id="tx-failed",
                pm_redeem_tx_state="STATE_FAILED",
                pm_settlement_status="redeem_failed",
            ),
            _record(
                condition_id=ALT_CONDITION_ID,
                pm_selected_token_id="456",
                pm_redeem_tx_id="tx-invalid",
                pm_redeem_tx_state="STATE_INVALID",
                pm_settlement_status="redeem_failed",
            ),
        ]

        candidates, _diagnostics = collect_redeem_candidates(
            positions,
            records,
            market_slug_prefix=MARKET_PREFIX,
            require_redeemable=True,
        )

        self.assertEqual({c["conditionId"] for c in candidates}, {CONDITION_ID, ALT_CONDITION_ID})

    def test_skip_losing_local_outcome_even_when_position_is_redeemable(self):
        candidates, diagnostics = collect_redeem_candidates(
            [_position(asset="123", redeemable=True)],
            [_record(actual_up=0, trade_side="yes", pm_selected_token_id="123")],
            market_slug_prefix=MARKET_PREFIX,
            require_redeemable=True,
        )

        self.assertEqual(candidates, [])
        self.assertIn("local_outcome_not_winning", {d["reason"] for d in diagnostics})

    def test_asset_mismatch_does_not_block_matching_winning_position(self):
        positions = [
            _position(asset="losing-token"),
            _position(asset="123"),
        ]

        candidates, diagnostics = collect_redeem_candidates(
            positions,
            [_record(actual_up=1, trade_side="yes", pm_selected_token_id="123")],
            market_slug_prefix=MARKET_PREFIX,
            require_redeemable=True,
        )

        self.assertEqual([c["asset"] for c in candidates], ["123"])
        self.assertIn(
            "position_asset_not_winning_record",
            {d["reason"] for d in diagnostics},
        )

    def test_require_redeemable(self):
        candidates, diagnostics = collect_redeem_candidates(
            [_position(redeemable=False)],
            [_record()],
            market_slug_prefix=MARKET_PREFIX,
            require_redeemable=True,
        )

        self.assertEqual(candidates, [])
        self.assertIn("not_redeemable", {d["reason"] for d in diagnostics})

    def test_build_redeem_transactions(self):
        specs = build_redeem_transactions(
            [{"conditionId": CONDITION_ID}],
            collateral_token_address=DEFAULT_POLYMARKET_PUSD_ADDRESS,
            ctf_address="0x4D97DCd97eC945f40cF65F87097ACe5EA0476045",
            target_address=DEFAULT_POLYMARKET_CTF_COLLATERAL_ADAPTER_ADDRESS,
            relayer_tx_type="SAFE",
        )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["conditionId"], CONDITION_ID)
        self.assertEqual(specs[0]["relayerTxType"], "SAFE")
        self.assertTrue(specs[0]["data"].startswith("0x01b7037c"))


if __name__ == "__main__":
    unittest.main()
