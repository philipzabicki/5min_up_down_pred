import json
import tempfile
import unittest
from pathlib import Path

from trade_policy import build_trade_intent, load_trade_policy_runtime_config


def _policy_payload(**overrides):
    payload = {
        "mode": "model_direction_min_stake",
        "submitted_price_mode": "entry_price_plus_ticks",
        "submitted_price_slippage_ticks": 3,
        "extra_buffer": 0.0,
        "stake_multiplier": 1.0,
        "fee_model": {
            "rate": 0.072,
            "exponent": 1.0,
            "fee_round_decimals": 5,
            "min_fee": 1e-05,
        },
    }
    payload.update(overrides)
    return payload


class TradePolicyRuntimeConfigTests(unittest.TestCase):
    def test_loads_entry_price_plus_ticks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.json"
            path.write_text(json.dumps(_policy_payload()), encoding="utf-8")

            cfg = load_trade_policy_runtime_config(path)

        self.assertEqual(cfg["submitted_price_mode"], "entry_price_plus_ticks")
        self.assertEqual(cfg["submitted_price_slippage_ticks"], 3)
        self.assertEqual(cfg["extra_buffer"], 0.0)
        self.assertEqual(cfg["stake_multiplier_mode"], "fixed")

    def test_loads_return_multiple_stake_multiplier_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.json"
            path.write_text(
                json.dumps(_policy_payload(stake_multiplier="return_multiple")),
                encoding="utf-8",
            )

            cfg = load_trade_policy_runtime_config(path)

        self.assertEqual(cfg["stake_multiplier"], 1.0)
        self.assertEqual(cfg["stake_multiplier_mode"], "return_multiple")

    def test_rejects_negative_slippage_ticks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.json"
            path.write_text(
                json.dumps(_policy_payload(submitted_price_slippage_ticks=-1)),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_trade_policy_runtime_config(path)

    def test_return_multiple_stake_uses_balance_return_and_rounds_amount(self):
        intent = build_trade_intent(
            policy_result={
                "decision": "buy_yes",
                "ask_yes": 0.5,
                "ask_no": 0.5,
            },
            bankroll=105.0,
            stake_multiplier=1.0,
            stake_multiplier_mode="return_multiple",
            initial_bankroll=100.0,
            return_multiple_balance=105.0,
            fee_model={
                "rate": 0.0,
                "exponent": 1.0,
                "fee_round_decimals": 5,
                "min_fee": 0.0,
            },
            order_min_size=3.0,
        )

        self.assertEqual(intent["final_reason"], "ok")
        self.assertAlmostEqual(intent["stake_multiplier"], 1.05)
        self.assertAlmostEqual(intent["required_stake_usdc"], 1.5)
        self.assertAlmostEqual(intent["effective_stake_usdc"], 1.58)
        self.assertAlmostEqual(intent["bet_usdc"], 1.58)


if __name__ == "__main__":
    unittest.main()
