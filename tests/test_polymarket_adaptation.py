import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from utils.polymarket_adaptation import SettlementCorrection, available_labels, decision_context
from utils.polymarket_policy import BASELINES, backtest, policy_action


START = pd.Timestamp('2026-01-01', tz='UTC')


def row(**changes):
    result = dict(condition_id='one', decision_available_at=START, timestamp_utc=START,
                  resolved_at_utc=START+pd.Timedelta(minutes=6), market_end_utc=START+pd.Timedelta(minutes=5),
                  p_model_up=.2, p_calibrated=.2, target_polymarket_up=0,
                  up_best_bid=.59, up_best_ask=.6, down_best_bid=.39, down_best_ask=.4,
                  up_ask_size=1000., down_ask_size=1000., order_min_size=5.,
                  seconds_to_expiry=300, quote_delay_ms=0, fee_rate=.07, fee_exponent=1,
                  fee_round_decimals=5, fee_min_fee=.00001, source='test', evaluation_fold=0)
    result.update(changes)
    return result


class AdaptationContractsTests(unittest.TestCase):
    def test_down_probability_and_exact_fee_expected_payout(self):
        config = dict(kind='fixed', stake=10, exact_fee_entry=True)
        action = policy_action(SimpleNamespace(**row()), .2, 100, 100, config, {})
        self.assertEqual(action['side'], 'down')
        self.assertAlmostEqual(action['p_side'], .8)
        self.assertAlmostEqual(action['fee'], .42)
        self.assertAlmostEqual(action['shares'], 9.58/.4)
        self.assertAlmostEqual(action['expected_pnl'], .8*9.58/.4-10)
        self.assertGreater(action['probability_edge'], 0)

    def test_additive_live_fee_is_more_conservative_than_exact_gross_fee(self):
        state = SimpleNamespace(**row())
        live = dict(mode='ev', extra_buffer=0., stake_multiplier=1., stake_multiplier_mode='fixed')
        # DOWN p=.43 exceeds .4/(1-.042), but not .4+.042.
        self.assertIsNone(policy_action(state, .57, 100, 100, BASELINES[1], live))
        exact = policy_action(state, .57, 100, 100, dict(kind='fixed', stake=10, exact_fee_entry=True), {})
        self.assertIsNotNone(exact)
        self.assertGreater(exact['expected_pnl'], 0)

    def test_independent_signal_keeps_betting_after_portfolio_bankruptcy(self):
        first = row(target_polymarket_up=1)
        second = row(condition_id='two', timestamp_utc=START+pd.Timedelta(minutes=10),
                     decision_available_at=START+pd.Timedelta(minutes=10),
                     market_end_utc=START+pd.Timedelta(minutes=15),
                     resolved_at_utc=START+pd.Timedelta(minutes=16), target_polymarket_up=1, evaluation_fold=1)
        data = pd.DataFrame([first, second])
        config = dict(kind='fixed', stake=100)
        portfolio, tr, _ = backtest(data, config, {})
        signal, independently_financed, _ = backtest(data, config, {}, independent_funding=True)
        self.assertEqual(len(tr), 1)
        self.assertEqual(portfolio['final_balance'], 0)
        self.assertEqual(len(independently_financed), 2)
        self.assertEqual(signal['net_pnl'], -200)
        self.assertIn('not a feasible', signal['funding'])
        data['down_ask_size'] = 1
        _, illiquid, _ = backtest(data, config, {}, independent_funding=True)
        self.assertTrue(illiquid.empty)

    def test_rejections_distinguish_liquidity_minimum_cash_and_locked(self):
        config = dict(kind='fixed', stake=10)
        cases = [(row(down_ask_size=1), 100, 100, 0, 'insufficient_liquidity'),
                 (row(order_min_size=100), 100, 100, 0, 'minimum_order'),
                 (row(), 5, 5, 0, 'insufficient_cash'),
                 (row(), 5, 100, 95, 'locked_capital'),
                 (row(), 0, 0, 0, 'no_cash')]
        for state, cash, equity, locked, expected in cases:
            action, reason = policy_action(SimpleNamespace(**state), .2, cash, equity, config, {}, locked, True)
            self.assertIsNone(action)
            self.assertEqual(reason, expected)

    def test_context_ignores_future_candles_and_handles_missing_minutes(self):
        t = pd.date_range(START, periods=50, freq='min')
        oof = pd.DataFrame({'Opened': t, 'Close': np.exp(np.arange(50)*.001)})
        before = decision_context(oof)
        oof.loc[40:, 'Close'] *= 10
        after = decision_context(oof)
        pd.testing.assert_frame_equal(before.iloc[:40], after.iloc[:40])
        self.assertEqual(before.context_available_at.iloc[30], t[30]+pd.Timedelta(minutes=1))
        missing = decision_context(oof.drop(index=35)).set_index('Opened')
        self.assertTrue(pd.isna(missing.loc[t[36], 'last_return_1m']))

    def test_labels_must_be_available_before_training_including_expiry(self):
        data = pd.DataFrame([row(resolved_at_utc=START-pd.Timedelta(minutes=1))])
        self.assertTrue(available_labels(data, START+pd.Timedelta(minutes=3)).empty)
        data = pd.DataFrame([row(resolved_at_utc=START+pd.Timedelta(minutes=8))])
        self.assertTrue(available_labels(data, START+pd.Timedelta(minutes=8)).empty)

    def test_correction_rejects_future_labels_and_unavailable_context(self):
        t = pd.date_range(START, periods=200, freq='5min')
        data = pd.DataFrame({'decision_available_at': t, 'market_end_utc': t+pd.Timedelta(minutes=5),
                             'resolved_at_utc': t+pd.Timedelta(minutes=6),
                             'context_available_at': t, 'p_model_up': np.linspace(.4, .6, 200),
                             'target_polymarket_up': np.arange(200)%2,
                             'past_volatility_30m': .001, 'last_return_1m': 0.})
        cutoff = t[-1]+pd.Timedelta(minutes=10)
        model = SettlementCorrection(1).fit(data, cutoff)
        self.assertTrue(np.isfinite(model.predict(data)).all())
        same = data.assign(up_best_ask=.99, down_best_ask=.01, underlying_move_bps=1000.)
        np.testing.assert_array_equal(model.predict(data), model.predict(same))
        with self.assertRaises(ValueError):
            SettlementCorrection(1).fit(data, t[-1])
        data.loc[0, 'context_available_at'] = t[0]+pd.Timedelta(seconds=1)
        with self.assertRaises(ValueError):
            model.predict(data)


if __name__ == '__main__':
    unittest.main()
