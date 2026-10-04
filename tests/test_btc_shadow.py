import json
import unittest

import run_btc_shadow as shadow


class BtcShadowTests(unittest.TestCase):
    def test_frozen_variants_produce_finite_probabilities_from_one_quote(self):
        weights = shadow._load_market_weights()
        quote = {
            "up_best_bid": 0.48,
            "up_best_ask": 0.49,
            "down_best_bid": 0.50,
            "down_best_ask": 0.51,
            "up_bid_size": 100.0,
            "up_ask_size": 100.0,
            "down_bid_size": 100.0,
            "down_ask_size": 100.0,
        }

        raw_btc, btc_platt, market_only, market_plus_btc = shadow._predict_variants(
            weights,
            0.61,
            quote,
        )

        self.assertEqual(raw_btc, 0.61)
        for probability in (btc_platt, market_only, market_plus_btc):
            self.assertGreater(probability, 0.0)
            self.assertLess(probability, 1.0)

    def test_protocol_predeclares_observation_count_and_hypothetical_fills(self):
        config = json.loads(shadow.PROTOCOL_PATH.read_text(encoding="utf-8"))

        self.assertEqual(config["portfolio"]["initial_virtual_usd"], 100.0)
        self.assertEqual(config["portfolio"]["fixed_hypothetical_stake_usd"], 5.0)
        self.assertEqual(
            config["predeclared_evaluation"]["checkpoint_hours"],
            [48, 168, 720],
        )
        self.assertTrue(config["predeclared_evaluation"]["no_automatic_extension"])
        previous = json.loads(
            (shadow.ROOT / "configs/btc_shadow_protocol_20261003.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            previous["predeclared_evaluation"]["minimum_observation_days"],
            730,
        )
        self.assertEqual(
            previous["predeclared_evaluation"]["extension_cap_days_after_start"],
            1095,
        )
        self.assertEqual(config["model_meta_path"], previous["model_meta_path"])
        self.assertEqual(config["model_weights_path"], previous["model_weights_path"])
        self.assertEqual(config["portfolio"], previous["portfolio"])
        self.assertEqual(
            config["execution_assumption"],
            previous["execution_assumption"],
        )
        self.assertFalse(config["execution_assumption"]["actual_fill_known"])
        self.assertFalse(config["execution_assumption"]["actual_fee_known"])

    def test_book_depth_uses_visible_size_at_best_price_and_top_five_levels(self):
        levels = [
            {"price": "0.40", "size": "4"},
            {"price": "0.41", "size": "6"},
            {"price": "0.40", "size": "3"},
            {"price": "0.42", "size": "2"},
            {"price": "0.43", "size": "1"},
            {"price": "0.44", "size": "7"},
        ]

        best_price, best_size, depth_shares, depth_usd, level_count = shadow._best_level(
            levels,
            "ask",
        )

        self.assertEqual(best_price, 0.40)
        self.assertEqual(best_size, 7.0)
        self.assertEqual(level_count, 5)
        self.assertEqual(depth_shares, 23.0)
        self.assertAlmostEqual(depth_usd, 9.61)

    def test_book_age_allows_small_future_source_clock_skew(self):
        for age in (-0.087153, -0.010567, 0.0, shadow.MAX_BOOK_AGE_SECONDS):
            with self.subTest(age=age):
                self.assertTrue(shadow._book_timestamp_age_is_acceptable(age))

        for age in (None, -1.001, shadow.MAX_BOOK_AGE_SECONDS + 0.001):
            with self.subTest(age=age):
                self.assertFalse(shadow._book_timestamp_age_is_acceptable(age))


if __name__ == "__main__":
    unittest.main()
