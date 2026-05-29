import json
import tempfile
import unittest
from pathlib import Path

from trade_policy import load_trade_policy_runtime_config


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

    def test_rejects_negative_slippage_ticks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.json"
            path.write_text(
                json.dumps(_policy_payload(submitted_price_slippage_ticks=-1)),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_trade_policy_runtime_config(path)


if __name__ == "__main__":
    unittest.main()
