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
from utils.polymarket_history import (PRIMARY, SECONDARY, combine_decisions, download_source, economic_dataset,
                                      join_oos, normalize_markets, normalize_quotes, payoff, reconcile_sources,
                                      secondary_adapter, select_quotes, select_secondary_quotes, token_mapping,
                                      utc, validate_oos, write_json)
from utils.polymarket_policy import (BASELINES, backtest, calibrate, chronological_splits, fit_calibrator,
                                     evaluate_walk_forward, optimal_kelly_stake, policy_action)
from build_polymarket_history import experiment_signature, load_current_oof

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
        validate_oos(p[['Opened', 'p_model_up']])  # accepted current training OOF, no invented provenance

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

    def portfolio_row(self, **changes):
        row = dict(condition_id='c', decision_available_at=START, timestamp_utc=START,
                   resolved_at_utc=START+pd.Timedelta(minutes=12), market_end_utc=START+pd.Timedelta(minutes=5),
                   p_model_up=.8, p_calibrated=.7, target_polymarket_up=1,
                   up_best_ask=.4, down_best_ask=.6, up_ask_size=1000., down_ask_size=1000.,
                   order_min_size=5., seconds_to_expiry=300, quote_delay_ms=0,
                   fee_rate=.25, fee_exponent=2, fee_round_decimals=5, fee_min_fee=.00001,
                   source=PRIMARY, evaluation_fold=0, selected_policy='fixed_5')
        row.update(changes)
        return row

    def test_oof_join_never_uses_a_later_or_wrong_minute(self):
        p = pd.DataFrame({'Opened': [START-pd.Timedelta(minutes=2), START+pd.Timedelta(minutes=4)],
                          'p_model_up': [.6, .9]})
        self.assertTrue(join_oos(p, self.normalize(), 0).empty)
        p.loc[len(p)] = [START-pd.Timedelta(minutes=1), .55]
        d = join_oos(p, self.normalize(), 1)
        self.assertEqual(len(d), 1)
        self.assertEqual(d.p_model_up.iloc[0], .55)
        self.assertEqual(d.condition_id.iloc[0], 'c')
        self.assertEqual(d.up_token_id.iloc[0], '111')
        self.assertEqual(d.decision_available_at.iloc[0], START+pd.Timedelta(seconds=1))

    def test_oof_rejects_duplicate_utc_and_invalid_probabilities(self):
        p = pd.DataFrame({'Opened': [str(START), '2026-03-25T00:10:00+02:00'], 'p_model_up': [.6, .7]})
        with self.assertRaises(ValueError): validate_oos(p)
        for probability in [np.nan, np.inf, -.01, 1.01]:
            with self.assertRaises(ValueError):
                validate_oos(pd.DataFrame({'Opened': [START], 'p_model_up': [probability]}))

    def test_source_priority_and_overlap_never_duplicate_transactions(self):
        one = pd.DataFrame([self.portfolio_row(eligible=True)])
        two = pd.DataFrame([self.portfolio_row(source=SECONDARY, eligible=True)])
        combined = combine_decisions(one, two)
        self.assertEqual(combined.source.tolist(), [PRIMARY])
        _, trades, _ = backtest(pd.concat([one, two]), {'kind': 'fixed', 'stake': 5}, {})
        self.assertEqual(len(trades), 1)
        one['eligible'] = False
        self.assertEqual(combine_decisions(one, two).source.tolist(), [SECONDARY])

    def test_overlapping_source_identity_or_settlement_conflict_is_rejected(self):
        one = self.normalize()
        self.assertEqual(reconcile_sources(one, one)['common_markets'], 1)
        for column, value in [('down_token_id', 'wrong'), ('polymarket_outcome_up', 0),
                              ('resolved_at_utc', START+pd.Timedelta(minutes=20))]:
            two = one.copy(); two[column] = value
            with self.assertRaises(ValueError): reconcile_sources(one, two)

    def test_fold_boundary_carries_locked_cash_and_original_settlement(self):
        rows = [self.portfolio_row(selected_policy='fixed_10_edge_02'),
                self.portfolio_row(condition_id='next', evaluation_fold=1, selected_policy='no_trade',
                    decision_available_at=START+pd.Timedelta(minutes=5), timestamp_utc=START+pd.Timedelta(minutes=5))]
        summary, trades, curve = backtest(pd.DataFrame(rows), {'kind': 'selected'}, {})
        self.assertEqual(len(trades), 1)
        boundary = curve[curve.timestamp_utc.eq(START+pd.Timedelta(minutes=5))].iloc[0]
        self.assertEqual(boundary.cash, 90)
        self.assertEqual(boundary.open_position_cost, 10)
        self.assertEqual(boundary.realized_pnl, 0)
        self.assertTrue(pd.isna(boundary.open_position_market_value))
        self.assertTrue(pd.isna(boundary.mark_to_market_equity))
        self.assertEqual(curve.iloc[-1].timestamp_utc, START+pd.Timedelta(minutes=12))
        self.assertAlmostEqual(summary['final_balance'], 100+trades.pnl.sum())
        self.assertAlmostEqual(curve.cash_flow.sum(), summary['net_pnl'])

    def test_locked_funds_cannot_be_spent_but_pending_positions_still_settle(self):
        rows = [self.portfolio_row(), self.portfolio_row(condition_id='next', evaluation_fold=1,
            decision_available_at=START+pd.Timedelta(minutes=5), timestamp_utc=START+pd.Timedelta(minutes=5))]
        summary, trades, curve = backtest(pd.DataFrame(rows), {'kind': 'fixed', 'stake': 100}, {})
        self.assertEqual(len(trades), 1)
        self.assertEqual(curve.iloc[1].cash, 0)
        self.assertEqual(curve.iloc[1].open_position_cost, 100)
        self.assertGreater(summary['minimum_order_unaffordable_decisions'], 0)
        self.assertGreater(summary['final_balance'], 100)
        self.assertEqual(summary['final_open_position_cost'], 0)
        rows[0]['target_polymarket_up'] = 0
        summary, _, _ = backtest(pd.DataFrame(rows), {'kind': 'fixed', 'stake': 100}, {})
        self.assertEqual(summary['net_pnl'], -100)
        self.assertEqual(summary['final_balance'], 0)

    def test_live_trade_records_the_raw_probability_used_by_run(self):
        config = {'mode': 'ev', 'extra_buffer': 0., 'stake_multiplier': 1., 'stake_multiplier_mode': 'fixed'}
        summary, trades, _ = backtest(pd.DataFrame([self.portfolio_row()]), BASELINES[1], config)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.probability.iloc[0], .8)
        self.assertEqual(trades.p_calibrated.iloc[0], .7)

    def test_minimum_order_unaffordability_uses_live_cent_rounding(self):
        row = self.portfolio_row(up_best_ask=.4, down_best_ask=.6, fee_rate=0.)
        # Net shares would fit in $1.999999, but the executable minimum is $2.00.
        summary, trades, _ = backtest(pd.DataFrame([row]), BASELINES[0], {}, initial_bankroll=1.999999)
        self.assertEqual(summary['minimum_order_unaffordable_decisions'], 1)
        self.assertEqual(len(trades), 0)

    def test_changed_oof_invalidates_probability_dependent_outputs(self):
        inventories = {PRIMARY: {'sha': 'primary'}, SECONDARY: {'sha': 'secondary'}}
        old, _ = experiment_signature({'sha256': 'old', 'metadata_sha256': 'meta'}, inventories)
        new, _ = experiment_signature({'sha256': 'new', 'metadata_sha256': 'meta'}, inventories)
        self.assertNotEqual(old, new)

    def test_missing_oof_reports_exact_path_without_a_substitute(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / 'current_btc_oof.parquet'
            with patch('build_polymarket_history.resolve_oof_prediction_output_paths', return_value={'parquet': missing}):
                with self.assertRaises(FileNotFoundError) as error:
                    load_current_oof()
            self.assertIn(str(missing.resolve()), str(error.exception))

    def test_missing_forward_quote_is_an_exclusion_not_an_imputed_fill(self):
        predictions = pd.DataFrame({'Opened': [START-pd.Timedelta(minutes=1)],
                                    'p_model_up': [.6], 'target_binance_proxy_up': [1]})
        q = normalize_quotes(quotes([START-pd.Timedelta(seconds=1)]), self.normalize())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'q.parquet'; q.to_parquet(path)
            joined = select_quotes(join_oos(predictions, self.normalize(), 0), path, 2000)
            result = economic_dataset(joined)
            self.assertFalse(result.eligible.iloc[0])
            self.assertEqual(result.exclusion_reason.iloc[0], 'missing_forward_quote')
            self.assertTrue(pd.isna(result.BUY_UP_5_pnl.iloc[0]))

    def test_secondary_quotes_are_forward_per_side_and_preserve_capture_times(self):
        data = pd.DataFrame({'condition_id': ['c']*4, 'timestamp_utc': [START-pd.Timedelta(milliseconds=1),
                            START+pd.Timedelta(milliseconds=100), START+pd.Timedelta(milliseconds=200),
                            START+pd.Timedelta(seconds=3)], 'token_side': ['up', 'up', 'down', 'down'],
                             'best_bid': .3, 'best_ask': .4, 'ask_size_0': 100., 'quote_valid': True})
        decisions = pd.DataFrame([{'decision_id': 'd', 'condition_id': 'c', 'decision_available_at': START,
                                  'market_end_utc': START+pd.Timedelta(minutes=5)}])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'books.parquet'; data.to_parquet(path, index=False)
            result = select_secondary_quotes(decisions, path, 2000)
            self.assertEqual(result.up_timestamp_utc.iloc[0], START+pd.Timedelta(milliseconds=100))
            self.assertEqual(result.timestamp_utc.iloc[0], START+pd.Timedelta(milliseconds=200))
            self.assertEqual(result.side_capture_gap_ms.iloc[0], 100)
            self.assertTrue(result.quote_valid.iloc[0])
            self.assertFalse(select_secondary_quotes(decisions, path, 150).quote_valid.iloc[0])

    def test_walk_forward_headline_uses_one_continuous_portfolio(self):
        rows = []
        for i in range(30):
            time = START+pd.Timedelta(minutes=5*i)
            rows.append(self.portfolio_row(condition_id=str(i), decision_available_at=time, timestamp_utc=time,
                market_end_utc=time+pd.Timedelta(minutes=5), resolved_at_utc=time+pd.Timedelta(minutes=12)))
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_walk_forward(pd.DataFrame(rows), tmp)
            tr = pd.read_parquet(Path(tmp)/'fixed_5_trades.parquet')
            curve = pd.read_parquet(Path(tmp)/'fixed_5_bankroll.parquet')
            headline = result['continuous_baselines']['fixed_5']
            self.assertEqual(headline['initial_bankroll'], 100)
            self.assertAlmostEqual(headline['final_balance'], 100+tr.pnl.sum())
            self.assertEqual(tr.fold_id.nunique(), 3)
            self.assertEqual((curve.event == 'settlement').sum(), len(tr))
            self.assertNotIn('aggregate_baselines', result)
            for summary in result['continuous_baselines'].values():
                self.assertEqual(summary['final_open_position_cost'], 0)
            for fold in result['folds']:
                self.assertLess(pd.Timestamp(fold['calibration_labels_available_before']), pd.Timestamp(fold['evaluation_start_utc']))


if __name__ == '__main__':
    unittest.main()
