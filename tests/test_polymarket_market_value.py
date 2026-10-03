import unittest

import numpy as np
import pandas as pd

from utils.polymarket_market_value import (MARKET_COLUMNS, _matrix, decide_at_observed_book,
    fit_market_model, market_features, simulate_fixed_policy)


START = pd.Timestamp('2026-08-01T00:00:00Z')


def market_row(index=0, **changes):
    start = START + pd.Timedelta(minutes=5*index)
    observed = start + pd.Timedelta(seconds=1)
    row = dict(condition_id=f'condition-{index}', market_slug=f'market-{index}',
        decision_id=f'decision-{index}', market_start_utc=start, timestamp_utc=observed,
        decision_available_at=start, second_layer_computed_at=observed,
        market_end_utc=start+pd.Timedelta(minutes=5),
        resolved_at_utc=start+pd.Timedelta(minutes=6), target_polymarket_up=1,
        p_model_up=.9, p_market_mid=.5, source_latency_s=0, fold=0,
        up_best_bid=.39, up_best_ask=.4, down_best_bid=.59, down_best_ask=.6,
        up_bid_size=100., down_bid_size=100., up_ask_size=100., down_ask_size=100.,
        fee_rate=.07, fee_exponent=1., fee_round_decimals=5, fee_min_fee=.00001,
        order_min_size=1., tick_size=.01, quote_delay_ms=0., seconds_to_expiry=299.,
        past_volatility_30m=.001, last_return_1m=.0001)
    row.update(changes)
    return row


def execution_quote(row, at, up_ask=.4, down_ask=.6):
    return pd.DataFrame([dict(condition_id=row['condition_id'],
        execution_quote_at=at, quote_valid=True,
        up_best_ask_quote=up_ask, down_best_ask_quote=down_ask,
        up_ask_size_quote=100., down_ask_size_quote=100.)]).set_index('condition_id')


class MarketValueContractsTests(unittest.TestCase):
    def test_features_use_only_observed_prices_and_share_quantities(self):
        row = market_row(target_polymarket_up=0, condition_id='secret-id')
        row.update(up_ask_size=9., down_ask_size=1., up_bid_size=5., down_bid_size=3.)
        features = market_features(pd.DataFrame([row]))
        self.assertEqual(list(features.columns), MARKET_COLUMNS)
        self.assertAlmostEqual(features.loc[0, 'up_mid'], .395)
        self.assertAlmostEqual(features.loc[0, 'p_market_mid'], .395/(.395+.595))
        self.assertAlmostEqual(features.loc[0, 'log_ask_size_up'], np.log1p(9.))
        self.assertNotIn('target_polymarket_up', features.columns)
        self.assertNotIn('condition_id', features.columns)

    def test_model_requires_available_labels_and_ignores_evaluation_labels(self):
        train = pd.DataFrame([market_row(i, target_polymarket_up=i % 2,
            resolved_at_utc=START-pd.Timedelta(minutes=1),
            market_end_utc=START-pd.Timedelta(minutes=2)) for i in range(12)])
        cutoff = START
        model = fit_market_model(train, 'market_only', 1., cutoff)
        test = pd.DataFrame([market_row(20), market_row(21, target_polymarket_up=0)])
        before = model.predict_proba(_matrix(test, 'market_only'))[:, 1]
        test['target_polymarket_up'] = 1-test.target_polymarket_up
        after = model.predict_proba(_matrix(test, 'market_only'))[:, 1]
        np.testing.assert_array_equal(before, after)
        late = train.copy()
        late.loc[0, 'resolved_at_utc'] = cutoff
        with self.assertRaisesRegex(ValueError, 'unavailable official label'):
            fit_market_model(late, 'market_only', 1., cutoff)

    def test_execution_delay_cannot_reselect_side_or_raise_observed_limit(self):
        row = market_row()
        decision, reasons = decide_at_observed_book(pd.Series(row), row['p_model_up'])
        self.assertFalse(reasons)
        self.assertEqual(decision[1], 'up')
        observed_limit = decision[2]
        self.assertEqual(observed_limit, .4)

        future = execution_quote(row, row['timestamp_utc']+pd.Timedelta(seconds=1),
                                 up_ask=.41, down_ask=.3)
        result, trades, _ = simulate_fixed_policy(pd.DataFrame([row]), 'p_model_up',
            execution_delay=1, bankroll=None, future_quotes=future)
        self.assertTrue(trades.empty)
        self.assertEqual(result['rejection_reasons']['execution_price_above_observed_limit'], 1)

    def test_delayed_fill_occurs_after_second_layer_timestamp(self):
        row = market_row()
        execution_at = row['timestamp_utc']+pd.Timedelta(seconds=1)
        future = execution_quote(row, execution_at, up_ask=.4)
        _, trades, _ = simulate_fixed_policy(pd.DataFrame([row]), 'p_model_up',
            execution_delay=1, bankroll=None, future_quotes=future)
        self.assertEqual(len(trades), 1)
        self.assertGreaterEqual(trades.execution_at.iloc[0], trades.second_layer_computed_at.iloc[0])
        self.assertEqual(trades.side.iloc[0], 'up')

    def test_fixed_stake_portfolio_carries_cash_through_settlement(self):
        data = pd.DataFrame([market_row(0, target_polymarket_up=1, fold=0),
                             market_row(1, target_polymarket_up=0, fold=1),
                             market_row(2, target_polymarket_up=1, fold=2)])
        result, trades, path = simulate_fixed_policy(data, 'p_model_up', bankroll=100.)
        self.assertEqual(len(trades), 3)
        self.assertAlmostEqual(result['final_balance'], 100.+trades.pnl.sum())
        self.assertEqual(result['initial_bankroll'], 100.)
        self.assertGreater(result['max_drawdown'], 0.)
        self.assertEqual(sum(x['trades'] for x in result['by_evaluation_fold'].values()), len(trades))
        self.assertEqual(set(result['by_evaluation_fold']), {'0', '1', '2'})
        self.assertTrue(path.cost_basis_equity.dropna().between(0, 110).all())

    def test_open_position_cash_stays_locked_until_market_resolution(self):
        data = pd.DataFrame([market_row(i, target_polymarket_up=1) for i in range(3)])

        result, trades, _ = simulate_fixed_policy(data, 'p_model_up', bankroll=5.)

        self.assertEqual(list(trades.condition_id), ['condition-0', 'condition-2'])
        self.assertEqual(result['rejection_reasons']['insufficient_cash'], 1)
        self.assertEqual(result['ending_locked_cost'], 0.)
        self.assertEqual(result['ending_open_positions'], 0)

    def test_observed_ask_size_and_minimum_order_are_enforced(self):
        row = pd.Series(market_row(up_ask_size=.1))
        decision, reasons = decide_at_observed_book(row, .9)
        self.assertIsNone(decision)
        self.assertIn('observed_liquidity', reasons)
        row = pd.Series(market_row(order_min_size=100.))
        decision, reasons = decide_at_observed_book(row, .9)
        self.assertIsNone(decision)
        self.assertIn('observed_minimum_order', reasons)

    def test_expected_pnl_buffer_preserves_cash_when_edge_is_too_small(self):
        row = pd.Series(market_row(p_model_up=.6))
        decision, reasons = decide_at_observed_book(row, .6)
        self.assertFalse(reasons)
        self.assertIsNotNone(decision)
        self.assertGreater(decision[0], 0.)

        buffer = decision[0] + .01
        result, trades, _ = simulate_fixed_policy(
            pd.DataFrame([row]),
            'p_model_up',
            bankroll=100.,
            min_expected_pnl=buffer,
        )

        self.assertTrue(trades.empty)
        self.assertEqual(result['rejection_reasons']['expected_pnl_below_buffer'], 1)
        self.assertEqual(result['ending_available_cash'], 100.)
        self.assertEqual(result['ending_locked_cost'], 0.)
        self.assertEqual(result['ending_open_positions'], 0)
        self.assertEqual(result['final_balance'], 100.)


if __name__ == '__main__':
    unittest.main()
