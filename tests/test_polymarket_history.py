import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa

from utils.polymarket import resolve_polymarket_actual_up_from_market_payload, polymarket_taker_fee_usdc_from_notional
from utils.polymarket_history import (download_source, economic_dataset, join_oos, normalize_markets, normalize_quotes,
                                      payoff, secondary_adapter, select_quotes, token_mapping, utc, validate_oos, write_json)
from utils.polymarket_policy import (BASELINES, backtest, calibrate, chronological_splits, fit_calibrator,
                                     optimal_kelly_stake, policy_action)

START = pd.Timestamp('2026-03-24T22:10:00Z')
FEES = {'rate': .25, 'exponent': 2, 'fee_round_decimals': 5, 'min_fee': .00001}


def market():
    return pd.DataFrame([{'condition_id': 'c', 'slug': 'btc-updown-5m-1774390200',
                          'token_up': '111', 'token_down': '222', 'market_start': START,
                          'market_end': START + pd.Timedelta(minutes=5), 'outcome': 'Down'}])


def payload():
    return {'conditionId': 'c', 'slug': 'btc-updown-5m-1774390200', 'outcomes': '["Down", "Up"]',
            'clobTokenIds': '["222", "111"]', 'outcomePrices': '["0", "1"]',
            'umaResolutionStatus': 'resolved', 'closed': True,
            'eventStartTime': str(START), 'endDate': str(START + pd.Timedelta(minutes=5)),
            'umaEndDate': str(START + pd.Timedelta(minutes=5, seconds=20)),
            'feesEnabled': True, 'feeSchedule': FEES, 'orderMinSize': 5, 'orderPriceMinTickSize': .01}


def quotes(times):
    return pd.DataFrame({'condition_id': 'c', 'ts_utc': times, 'bu': .4, 'au': .45,
                         'bd': .5, 'ad': .55, 'su': 100., 'sd': 100., 'sau': 100., 'sad': 100.,
                         'du': 1000., 'dd': 1000.})


class HistoricalContractsTests(unittest.TestCase):
    def normalize(self, p=None):
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp) / (market().slug.iloc[0] + '.json'), {'payload': payload() if p is None else p})
            return normalize_markets(market(), tmp)

    def test_token_order_is_label_based(self):
        self.assertEqual(token_mapping(payload()), {'down': '222', 'up': '111'})
        p = payload(); p['outcomes'] = '["Up", "Up"]'
        with self.assertRaises(ValueError): token_mapping(p)

    def test_utc_normalization(self):
        self.assertEqual(utc('2026-03-25T00:10:00+02:00'), START)
        self.assertEqual(utc('2026-03-24 22:10:00'), START)
        times = pd.Series([START]).astype('datetime64[us, UTC]')
        self.assertEqual(str(utc(times).dtype), 'datetime64[ns, UTC]')

    def test_official_settlement_not_vendor_inference(self):
        m = self.normalize()
        self.assertEqual(m.polymarket_outcome_up.iloc[0], 1)
        self.assertEqual(m.vendor_inferred_outcome.iloc[0], 'Down')
        self.assertEqual(m.fee_rate.iloc[0], .25)
        self.assertEqual(resolve_polymarket_actual_up_from_market_payload(payload()), 1)

    def test_unresolved_even_terminal_book_is_rejected(self):
        p = payload(); p['umaResolutionStatus'] = ''; p['closed'] = False
        m = self.normalize(p)
        self.assertTrue(pd.isna(m.polymarket_outcome_up.iloc[0]))

    def test_ambiguous_resolved_prices_are_not_settlement(self):
        p = payload(); p['outcomePrices'] = '["0.1", "0.9"]'
        self.assertTrue(pd.isna(self.normalize(p).polymarket_outcome_up.iloc[0]))

    def test_label_availability_uses_later_official_time(self):
        p = payload(); p['closedTime'] = str(START + pd.Timedelta(minutes=10))
        self.assertEqual(self.normalize(p).resolved_at_utc.iloc[0], START + pd.Timedelta(minutes=10))

    def test_market_alignment_uses_trading_start(self):
        p = payload(); p['startDate'] = '2026-03-23T12:00:00Z'
        self.assertEqual(self.normalize(p).validation_status.iloc[0], 'validated')
        p['eventStartTime'] = str(START + pd.Timedelta(seconds=1))
        self.assertFalse(self.normalize(p).token_mapping_valid.iloc[0])

    def test_missing_crossed_and_invalid_books(self):
        raw = quotes([START, START + pd.Timedelta(seconds=1), START + pd.Timedelta(seconds=2)])
        raw.loc[0, 'au'] = np.nan; raw.loc[1, 'bu'] = .8; raw.loc[2, 'ad'] = 1.2
        q = normalize_quotes(raw, self.normalize())
        self.assertTrue(q.missing_side.iloc[0]); self.assertTrue(q.crossed_book.iloc[1])
        self.assertTrue(q.invalid_price.iloc[2]); self.assertFalse(q.quote_valid.any())

    def test_duplicate_and_nonmonotonic_ticks(self):
        raw = quotes([START + pd.Timedelta(seconds=1), START, START])
        q = normalize_quotes(raw, self.normalize())
        self.assertTrue(q.non_monotonic_timestamp.iloc[1])
        self.assertEqual(q.duplicate_timestamp.sum(), 2)

    def test_no_quote_before_decision_can_ever_execute(self):
        # The nearest quote is earlier by 1ms; forward quote is later by 999ms.
        decision = START + pd.Timedelta(milliseconds=1)
        q = normalize_quotes(quotes([START, START + pd.Timedelta(seconds=1)]), self.normalize())
        d = pd.DataFrame([{'decision_id': 'd', 'condition_id': 'c', 'decision_available_at': decision}])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'q.parquet'; q.to_parquet(p)
            joined = select_quotes(d, p, 1000)
            self.assertEqual(joined.timestamp_utc.iloc[0], START + pd.Timedelta(seconds=1))
            self.assertEqual(joined.quote_delay_ms.iloc[0], 999)
            rejected = select_quotes(d, p, 500)
            self.assertTrue(pd.isna(rejected.timestamp_utc.iloc[0]))

    def test_first_invalid_quote_is_not_silently_skipped(self):
        raw = quotes([START, START + pd.Timedelta(seconds=1)])
        raw.loc[0, 'bu'] = .8
        q = normalize_quotes(raw, self.normalize())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'q.parquet'; q.to_parquet(p)
            d = pd.DataFrame([{'decision_id': 'd', 'condition_id': 'c', 'decision_available_at': START}])
            self.assertFalse(select_quotes(d, p, 1000).quote_valid.iloc[0])

    def test_forward_join_across_batches_keeps_first_quote(self):
        m = self.normalize()
        one = pa.Table.from_pandas(normalize_quotes(quotes([START + pd.Timedelta(seconds=1)]), m), preserve_index=False)
        two = pa.Table.from_pandas(normalize_quotes(quotes([START + pd.Timedelta(seconds=2)]), m), preserve_index=False)
        d = pd.DataFrame([{'decision_id': 'd', 'condition_id': 'c', 'decision_available_at': START}])
        with patch('utils.polymarket_history.pq.ParquetFile') as f:
            f.return_value.iter_batches.return_value = [one, two]
            result = select_quotes(d, 'unused', 2000)
        self.assertEqual(result.timestamp_utc.iloc[0], START + pd.Timedelta(seconds=1))

    def test_duplicate_batch_boundary_is_flagged(self):
        q = normalize_quotes(quotes([START]), self.normalize(), previous={'c': START})
        self.assertTrue(q.duplicate_timestamp.iloc[0])

    def test_oos_join_exact_live_row(self):
        p = pd.DataFrame([{'Opened': START - pd.Timedelta(minutes=1), 'p_model_up': .6,
                           'fold_id': 1, 'source_model_id': 'm', 'is_oos': True,
                           'fit_labels_available_at': START - pd.Timedelta(days=1)}])
        d = join_oos(p, self.normalize(), 2)
        self.assertEqual(d.decision_available_at.iloc[0], START + pd.Timedelta(seconds=2))
        p.loc[0, 'fit_labels_available_at'] = START
        with self.assertRaises(ValueError): validate_oos(p)
        with self.assertRaises(ValueError): validate_oos(p.drop(columns='fold_id'))

    def test_economic_buy_up_down_fee_and_depth(self):
        for won in (0, 1):
            result = payoff(10, .4, won, FEES, 100)
            fee = polymarket_taker_fee_usdc_from_notional(10, .4, FEES)['fee_usdc']
            self.assertAlmostEqual(result['pnl'], (10-fee)/.4*won-10)
        self.assertIsNone(payoff(10, .4, 1, FEES, 1))
        self.assertIsNone(payoff(10, .4, np.nan, FEES, 100))

    def test_counterfactual_targets_and_no_trade(self):
        m = self.normalize()
        p = pd.DataFrame([{'Opened': START-pd.Timedelta(minutes=1), 'p_model_up': .6,
                          'target_binance_proxy_up': 0, 'fold_id': 0, 'source_model_id': 'm',
                          'is_oos': True, 'fit_labels_available_at': START-pd.Timedelta(days=1)}])
        q = normalize_quotes(quotes([START]), m)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'q.parquet'; q.to_parquet(path)
            d = economic_dataset(select_quotes(join_oos(p, m, 0), path, 1000))
        self.assertEqual(d.NO_TRADE_pnl.iloc[0], 0)
        self.assertGreater(d.BUY_UP_5_pnl.iloc[0], 0)
        self.assertEqual(d.BUY_DOWN_5_pnl.iloc[0], -5)
        self.assertFalse(d.BUY_UP_100_available.iloc[0])
        self.assertTrue(d.target_mismatch.iloc[0])

    def test_past_only_calibration_and_walk_forward(self):
        t = pd.date_range('2026-01-01', periods=2000, freq='5min', tz='UTC')
        d = pd.DataFrame({'decision_available_at': t, 'resolved_at_utc': t + pd.Timedelta(minutes=6),
                          'target_polymarket_up': np.arange(2000) % 2, 'p_model_up': .6})
        for fold, past, future in chronological_splits(d):
            self.assertLess(past.resolved_at_utc.max(), future.decision_available_at.min())
            model = fit_calibrator(past, 'platt', future.decision_available_at.min())
            self.assertTrue(np.isfinite(calibrate(model, future.p_model_up)).all())
        with self.assertRaises(ValueError): fit_calibrator(d, 'platt', t[100])
        isotonic = fit_calibrator(d, 'isotonic', t[-1] + pd.Timedelta(days=1))
        self.assertTrue(np.isfinite(calibrate(isotonic, [.1, .9])).all())

    def test_secondary_preserves_token_identity_and_depth(self):
        raw = pd.DataFrame([{'asset': 'BTC', 'market_id': 'btc-updown-5m-1774390200',
                             'condition_id': 'c', 'timestamp': START, 'token_id': '111',
                             'best_bid': .4, 'best_ask': .45,
                             'bid_levels': '[{"price":0.3,"size":10},{"price":0.4,"size":20}]',
                             'ask_levels': '[{"price":0.6,"size":30},{"price":0.45,"size":40}]'}])
        q = secondary_adapter(raw, self.normalize())
        self.assertEqual(q.token_side.iloc[0], 'up')
        self.assertEqual(q.ask_price_0.iloc[0], .45)
        self.assertEqual(q.ask_size_0.iloc[0], 40)
        self.assertEqual(q.bid_price_0.iloc[0], .4)
        self.assertNotIn('ask_levels', q)

    def test_resumable_idempotent_download_and_checksum(self):
        body = b'firstsecond'
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'vendor' / 'revision'; directory.mkdir(parents=True)
            (directory / 'README.part').write_bytes(b'first')
            response = MagicMock(status_code=206, headers={'Content-Range': 'bytes 5-10/11'})
            response.__enter__.return_value = response
            response.iter_content.return_value = [b'second']
            with patch('utils.polymarket_history.session') as make:
                make.return_value.get.return_value = response
                download_source('vendor/dataset', ['README'], tmp, 'revision')
                self.assertEqual(make.return_value.get.call_args.kwargs['headers'], {'Range': 'bytes=5-'})
            self.assertEqual((directory / 'README').read_bytes(), body)
            with patch('utils.polymarket_history.session') as make:
                download_source('vendor/dataset', ['README'], tmp, 'revision')
                make.return_value.get.assert_not_called()
            (directory / 'README').write_bytes(b'corrupt')
            with self.assertRaises(ValueError): download_source('vendor/dataset', ['README'], tmp, 'revision')

    def test_kelly_uses_fee_adjusted_payoff(self):
        stake = optimal_kelly_stake(.7, .5, FEES, 100, 10000, 5)
        self.assertGreater(stake, 0)
        self.assertLess(stake, 40)  # fee-free full Kelly is $40

    def test_settlement_locks_stake_and_one_trade_per_market(self):
        row = dict(condition_id='c', decision_available_at=START, timestamp_utc=START,
                   resolved_at_utc=START+pd.Timedelta(minutes=6), market_end_utc=START+pd.Timedelta(minutes=5),
                   p_calibrated=.8, target_polymarket_up=1, up_best_ask=.4, down_best_ask=.6,
                   up_ask_size=1000., down_ask_size=1000., order_min_size=5.,
                   seconds_to_expiry=300, quote_delay_ms=0, fee_rate=.25, fee_exponent=2,
                   fee_round_decimals=5, fee_min_fee=.00001)
        data = pd.DataFrame([row, dict(row, decision_available_at=START+pd.Timedelta(seconds=1))])
        # The entry policy must operate without any settlement label in its state.
        state = SimpleNamespace(**{k: v for k, v in row.items() if k != 'target_polymarket_up'})
        action = policy_action(state, .8, 100, 100, {'kind': 'fixed', 'stake': 5}, {})
        self.assertIsNotNone(action)
        self.assertNotIn('pnl', action)
        summary, trades, curve = backtest(data, {'kind': 'fixed', 'stake': 5}, {})
        self.assertEqual(len(trades), 1)
        self.assertEqual(curve.iloc[0].cash, 95)
        self.assertEqual(curve.iloc[0].bankroll, 100)
        self.assertEqual(curve.iloc[-1].exposure, 0)
        self.assertGreater(summary['net_pnl'], 0)
        summary, _, _ = backtest(data, BASELINES[0], {})
        self.assertEqual(summary['net_pnl'], 0)


if __name__ == '__main__':
    unittest.main()
