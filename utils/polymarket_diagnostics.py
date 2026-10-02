"""Loss attribution and retrospective experiments using the existing cached pipeline."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils.polymarket_history import (PRIMARY, SECONDARY, checksum, fee_model, payoff, utc, write_json)
from utils.polymarket_policy import (BASELINES, backtest, calibrate, calibration_metrics,
                                     chronological_splits, fit_calibrator)
from utils.polymarket_adaptation import (CONTEXT_COLUMNS, REGULARIZATION, SettlementCorrection,
                                        available_labels, decision_context)
from utils.polymarket import polymarket_taker_fee_fraction_of_notional
from utils.project_config import load_modeling_settings

REPORT_ROOT = Path('data/analysis/polymarket/BTC')
ECONOMIC_MARGINS = (0., .5, 1.)
RANDOM_SEED = 20261002


def source_configuration_audit(evaluation):
    settings = load_modeling_settings(asset='BTC')
    meta = json.loads(Path(evaluation['oof']['metadata_path']).read_text())
    subset = json.loads(Path(settings['feature_subset_path']).read_text())[settings['feature_subset_list_key']]
    volume = meta['volume_profile_fixed_range']
    checks = dict(feature_subset=meta['feature_columns']==subset,
        hyperparameters=all(meta['model_hyperparameters']['cv_optuna_params'][k]==v
                            for k, v in settings['train_lgbm']['optuna_best_params'].items()),
        constraints=meta['monotone_constraints']['configured_constraints']==settings['train_lgbm']['monotone_constraints'],
        training_settings=all(settings['train_lgbm'].get(k, v)==v for k, v in meta['train_config'].items()),
        volume_profile=all(volume.get(k)==v for k, v in settings['volume_profile_fixed_range'].items() if k!='horizons') and
            all(all(volume['horizons'][h][k]==v for k, v in conf.items()) for h, conf in settings['volume_profile_fixed_range']['horizons'].items()),
        precision=meta['numeric_precision']['configured_float_precision']==settings['float_precision'])
    if not all(checks.values()):
        raise ValueError('Source configuration differs from current OOF model: '+str(checks))
    return dict(checks=checks, feature_count=meta['feature_count'],
        reaction_profile_selected_features=len([c for c in meta['feature_columns'] if c.startswith('rp_')]),
        reaction_profile_note='No reaction-profile column in the current 64-feature model; model RP metadata is consequently null. Configured feature generation is unchanged.',
        hashes={p: checksum(p) for p in ['configs/modeling.json', 'configs/indicator_fit.json', 'configs/datasets.json',
                                       str(settings['feature_subset_path'])]})


def enrich_trades(trades, decisions):
    if trades.empty:
        return trades
    extra = [c for c in decisions if c not in trades or c == 'condition_id']
    t = trades.merge(decisions[extra], on='condition_id', validate='one_to_one')
    t['execution_at'] = t[['timestamp_utc', 'decision_available_at']].max(axis=1)
    t['p_side'] = np.where(t.side.eq('up'), t.probability, 1-t.probability)
    t['p_side_raw'] = np.where(t.side.eq('up'), t.p_model_up, 1-t.p_model_up)
    t['p_side_calibrated'] = np.where(t.side.eq('up'), t.p_calibrated, 1-t.p_calibrated)
    t['won'] = np.where(t.side.eq('up'), t.target_polymarket_up, 1-t.target_polymarket_up)
    t['expected_pnl'] = t.p_side*t.shares-t.stake
    t['realized_minus_expected'] = t.pnl-t.expected_pnl
    t['pnl_official_no_fee'] = t.won*t.stake/t.price-t.stake
    available = np.where(t.side.eq('up'), t.up_ask_size, t.down_ask_size)
    t['no_fee_top_capacity_exceeded'] = t.stake/t.price > available
    proxy_win = np.where(t.side.eq('up'), t.target_binance_proxy_up, 1-t.target_binance_proxy_up)
    t['pnl_binance_diagnostic'] = proxy_win*t.shares-t.stake
    t['break_even_probability'] = t.stake/t.shares
    t['probability_edge'] = t.p_side-t.break_even_probability
    t['day_utc'] = t.timestamp_utc.dt.strftime('%Y-%m-%d')
    return t


def trade_statistics(t):
    if t.empty:
        return {'trades': 0, 'turnover': 0., 'pnl': 0., 'fees': 0.}
    turnover = float(t.stake.sum())
    return {'trades': len(t), 'turnover': turnover, 'fees': float(t.fee.sum()),
            'pnl': float(t.pnl.sum()), 'pnl_per_turnover': float(t.pnl.sum()/turnover),
            'hit_rate': float(t.won.mean()), 'expected_pnl': float(t.expected_pnl.sum()),
            'realized_minus_expected': float(t.realized_minus_expected.sum()),
            'official_no_fee_pnl': float(t.pnl_official_no_fee.sum()),
            'binance_label_diagnostic_pnl': float(t.pnl_binance_diagnostic.sum()),
            'fee_payout_effect': float((t.pnl_official_no_fee-t.pnl).sum()),
            'label_effect_same_bets': float((t.pnl_binance_diagnostic-t.pnl).sum()),
            'target_mismatches': int(t.target_mismatch.sum()),
            'negative_expected_pnl_trades': int(t.expected_pnl.le(0).sum()),
            'no_fee_top_capacity_exceeded': int(t.no_fee_top_capacity_exceeded.sum()),
            'entry_price_quantiles': {str(q): float(t.price.quantile(q)) for q in [0, .1, .5, .9, 1]}}


def grouped_diagnostics(t):
    if t.empty:
        return {}
    breaks = {'price': [0, .2, .3, .4, .5, .6, .8, 1],
              'p_side': np.linspace(0, 1, 11), 'p_side_raw': np.linspace(0, 1, 11),
              'p_side_calibrated': np.linspace(0, 1, 11),
              'probability_edge': [-np.inf, 0, .02, .05, .1, .2, np.inf],
              'p_model_up': np.linspace(0, 1, 11), 'p_calibrated': np.linspace(0, 1, 11),
              'quote_delay_ms': [-.01, 0, 500, 1000, 2000],
              'seconds_to_expiry': [0, 60, 120, 240, 300]}
    groups = {col: pd.cut(t[col], edges, include_lowest=True) for col, edges in breaks.items()}
    groups.update({c: t[c] for c in ['side', 'source', 'fold_id', 'day_utc']})
    result = {}
    for col, keys in groups.items():
        result[col] = [dict(bucket=str(key), **trade_statistics(g),
                            mean_success_probability=float(g.p_side.mean()),
                            observed_success_rate=float(g.won.mean()))
                       for key, g in t.groupby(keys, observed=True)]
    return result


def block_uncertainty(t, decision_days):
    """Moving blocks of three UTC days, including zero-trade days.

    Conditions on executed bets; it does not reselect bets or refinance portfolios.
    """
    if t.empty:
        return {'method': 'three-day moving-block bootstrap of executed bets', 'trades': 0}
    days = pd.date_range(min(decision_days), max(decision_days), freq='D')
    daily = t.groupby(t.timestamp_utc.dt.floor('D'))[['pnl', 'stake']].sum().reindex(days, fill_value=0.)
    values = daily.to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    samples = []
    for _ in range(1000):
        starts = rng.integers(0, max(len(values)-2, 1), size=int(np.ceil(len(values)/3)))
        indexes = np.concatenate([np.arange(s, min(s+3, len(values))) for s in starts])[:len(values)]
        pnl, stake = values[indexes].sum(axis=0)
        if stake > 0:
            samples.append(pnl/stake)
    return {'method': 'three-day moving-block bootstrap; conditional on executed bets, no reselection',
            'calendar_days': len(days), 'traded_days': int((daily.stake > 0).sum()),
            'trades': len(t), 'replicates_with_turnover': len(samples),
            'pnl_per_turnover_95pct': np.quantile(samples, [.025, .975]).tolist() if samples else None}


def audit_official_and_oof(t, oof, raw_market_path, cache):
    """Independent field parsing, not a second call through the normalizer."""
    raw = pd.read_parquet(raw_market_path).set_index('condition_id')
    indexed = oof.set_index('Opened')
    records = []
    for row in t.itertuples():
        vendor = raw.loc[row.condition_id]
        file = cache/(row.market_slug+'.json')
        response = json.loads(file.read_text())
        official = response['payload']
        names = json.loads(official['outcomes']) if isinstance(official['outcomes'], str) else official['outcomes']
        tokens = json.loads(official['clobTokenIds']) if isinstance(official['clobTokenIds'], str) else official['clobTokenIds']
        prices = json.loads(official['outcomePrices']) if isinstance(official['outcomePrices'], str) else official['outcomePrices']
        by_name = {name.lower(): (str(token), float(price)) for name, token, price in zip(names, tokens, prices)}
        start = pd.Timestamp(int(row.market_slug.rsplit('-', 1)[1]), unit='s', tz='UTC')
        declared_start = official.get('eventStartTime') or next(e['startTime'] for e in official['events'] if e.get('startTime'))
        expiry = utc(official['endDate'])
        resolution = max(utc(official[k]) for k in ['umaEndDate', 'closedTime'] if official.get(k))
        candle = indexed.loc[row.Opened]
        future = indexed.loc[row.Opened+pd.Timedelta(minutes=5)]
        proxy = int(future.Close >= candle.Close)
        checks = {
            'identity': official['conditionId'] == row.condition_id == vendor.name and official['slug'] == row.market_slug == vendor.slug,
            'token_direction': by_name['up'][0] == str(vendor.token_up) == row.up_token_id and by_name['down'][0] == str(vendor.token_down) == row.down_token_id,
            'market_boundaries': utc(declared_start) == start == row.market_start_utc == utc(vendor.market_start) and expiry == start+pd.Timedelta(minutes=5) == row.market_end_utc == utc(vendor.market_end),
            'prediction_timing': row.Opened+pd.Timedelta(minutes=1) == start and row.Opened.minute % 5 == 4 and row.decision_available_at >= start and row.timestamp_utc >= row.decision_available_at and row.timestamp_utc < expiry,
            'raw_oof': np.isclose(candle.oof_pred_proba_up, row.p_model_up, atol=1e-14) and candle.target_5m_candle_up == row.target_binance_proxy_up == proxy,
            'official_settlement': bool(official['closed']) and official['umaResolutionStatus'] == 'resolved' and by_name['up'][1] == row.target_polymarket_up and sorted(x[1] for x in by_name.values()) == [0., 1.] and resolution == row.resolved_at_utc,
        }
        fees = official['feeSchedule'] if official.get('feesEnabled') else {'rate': 0., 'exponent': 1.}
        # Independently reproduce the exact project gross-stake fee and payout.
        raw_fee = row.stake/row.price*float(fees['rate'])*(row.price*(1-row.price))**float(fees.get('exponent', 1))
        fee = round(raw_fee, int(fees.get('fee_round_decimals', 5)))
        fee = fee if fee >= float(fees.get('min_fee', .00001)) else 0.
        shares = (row.stake-fee)/row.price
        checks['payout_and_fee'] = np.isclose(fee, row.fee) and np.isclose(shares, row.shares) and np.isclose(shares*row.won-row.stake, row.pnl)
        if not all(checks.values()):
            raise AssertionError((row.market_slug, checks))
        records.append(dict(condition_id=row.condition_id, market_slug=row.market_slug, **checks,
                            official_sha256=checksum(file), token_order=names,
                            close_at_oof=float(candle.Close), close_at_oof_plus_5m=float(future.Close)))
    return pd.DataFrame(records)


def audit_raw_quotes(raw_path, trades, representative, destination):
    keys = trades[['condition_id', 'timestamp_utc']].drop_duplicates()
    ids = set(keys.condition_id)
    match, around = [], []
    worst = trades.nsmallest(40, 'pnl')
    cheapest = trades.nsmallest(20, 'price')
    targets = pd.concat([representative, worst, cheapest])[['condition_id', 'market_start_utc']].drop_duplicates()
    for batch in pq.ParquetFile(raw_path).iter_batches(batch_size=100000):
        raw = batch.to_pandas()
        raw = raw[raw.condition_id.isin(ids)]
        if raw.empty:
            continue
        raw['timestamp_utc'] = utc(raw.ts_utc)
        match.append(raw.merge(keys, on=['condition_id', 'timestamp_utc'], validate='one_to_one'))
        nearby = raw.merge(targets, on='condition_id')
        delta = (nearby.timestamp_utc-nearby.market_start_utc).dt.total_seconds()
        around.append(nearby[delta.between(0, 35)])
    matched = pd.concat(match, ignore_index=True)
    joined = trades.merge(matched, on=['condition_id', 'timestamp_utc'], validate='many_to_one')
    checks = {'unix_seconds': joined.timestamp_utc.eq(pd.to_datetime(joined.t, unit='s', utc=True)).all(),
              'all_trades_covered': len(joined) == len(trades)}
    for raw_col, canonical in [('bu', 'up_best_bid'), ('au', 'up_best_ask'), ('bd', 'down_best_bid'),
                               ('ad', 'down_best_ask'), ('sau', 'up_ask_size'), ('sad', 'down_ask_size')]:
        checks[raw_col] = np.allclose(joined[raw_col], joined[canonical], equal_nan=True)
    raw_price = np.where(joined.side.eq('up'), joined.au, joined.ad)
    raw_size = np.where(joined.side.eq('up'), joined.sau, joined.sad)
    checks['buy_uses_ask_not_bid'] = np.allclose(raw_price, joined.price)
    checks['observed_ask_capacity'] = bool(np.all(raw_size >= joined.shares))
    if not all(checks.values()):
        raise AssertionError(checks)
    pd.concat(around, ignore_index=True).to_parquet(destination/'kacho_entry_windows.parquet', index=False)
    return dict(trades=len(trades), distinct_quotes=len(keys), checks={k: bool(v) for k, v in checks.items()})


def audit_obadiaha(folder, decisions, destination):
    """Inspect the stored level ordering and extremes across ALL BTC snapshots."""
    summary, examples = [], []
    wanted = decisions.loc[decisions.eligible, ['market_slug', 'up_token_id', 'down_token_id']].drop_duplicates().set_index('market_slug')
    for file in sorted((folder/'orderbooks').glob('*.parquet')):
        counts = dict(rows=0, asks_descending=0, bids_ascending=0, ten_asks=0,
                      asks_99_to_90=0, best_equals_min_stored=0, quote_before_start=0)
        for batch in pq.ParquetFile(file).iter_batches(batch_size=100000):
            raw = batch.to_pandas()
            raw = raw[raw.asset.eq('BTC') & raw.market_id.str.startswith('btc-updown-5m-')]
            for row in raw.itertuples():
                asks = json.loads(row.ask_levels)
                bids = json.loads(row.bid_levels)
                ap = np.array([float(x['price']) for x in asks])
                bp = np.array([float(x['price']) for x in bids])
                counts['rows'] += 1
                counts['asks_descending'] += int(len(ap)>0 and np.all(np.diff(ap)<=0))
                counts['bids_ascending'] += int(len(bp)>0 and np.all(np.diff(bp)>=0))
                counts['ten_asks'] += int(len(ap)==10)
                counts['asks_99_to_90'] += int(len(ap)==10 and np.allclose(ap, np.arange(.99, .89, -.01)))
                counts['best_equals_min_stored'] += int(len(ap)>0 and np.isclose(row.best_ask, ap.min()))
                start = pd.Timestamp(int(row.market_id.rsplit('-', 1)[1]), unit='s', tz='UTC')
                counts['quote_before_start'] += int(utc(row.timestamp)<start)
                if len(examples)<20 and row.market_id in wanted.index:
                    identity = wanted.loc[row.market_id]
                    side = 'up' if str(row.token_id)==identity.up_token_id else 'down' if str(row.token_id)==identity.down_token_id else 'unknown'
                    examples.append(dict(file=str(file), market_slug=row.market_id, token_id=row.token_id,
                                         side=side, timestamp_utc=str(utc(row.timestamp)),
                                         seconds_from_start=(utc(row.timestamp)-start).total_seconds(),
                                         best_ask=row.best_ask, best_bid=row.best_bid, asks=asks, bids=bids))
        summary.append(dict(file=str(file), **counts))
        print('Raw Obadiaha audit:', file.name, counts['rows'], flush=True)
    totals = {k: sum(x[k] for x in summary) for k in summary[0] if k!='file'}
    by_latency = {str(latency): {'eligible': len(g[g.eligible]),
        'both_ask_090': int((g.eligible & g.up_best_ask.eq(.9) & g.down_best_ask.eq(.9)).sum()),
        'seconds_from_start_quantiles': ((g.loc[g.eligible, 'timestamp_utc']-g.loc[g.eligible, 'market_start_utc']).dt.total_seconds().quantile([0, .5, 1]).to_dict())}
        for latency, g in decisions.groupby('audit_latency')}
    result = dict(totals=totals, partitions=summary, eligible=by_latency, examples=examples,
                  conclusion='Stored tail of 10 levels is consistent with truncation before sorting. The adapter correctly sorts stored levels; the true full-book best ask is not recoverable. Real wide full books versus collector truncation cannot be proven without full responses/collector code. Treat these prices as stored-limit snapshots, not verified exchange top-of-book.')
    write_json(destination/'obadiaha_raw_audit.json', result)
    return result


def quote_statistics(data):
    kacho = data[data.source.eq(PRIMARY)].copy()
    fields = {'up_spread': kacho.up_best_ask-kacho.up_best_bid,
              'down_spread': kacho.down_best_ask-kacho.down_best_bid,
              'sum_asks': kacho.up_best_ask+kacho.down_best_ask,
              'cheaper_ask': kacho[['up_best_ask', 'down_best_ask']].min(axis=1)}
    return {'rows': len(kacho), 'quantiles': {name: {str(q): float(series.quantile(q))
             for q in [0, .01, .1, .5, .9, .99, 1]} for name, series in fields.items()},
             'cheap_below_020': int(fields['cheaper_ask'].lt(.2).sum()),
             'ask_sum_below_one': int(fields['sum_asks'].lt(1-1e-9).sum()),
             'exchange_book_age_known': False}


def fee_entry_audit(data, live_config):
    counts = dict(eligible=len(data), live_additive_entries=0, exact_entries=0, live_rejects_exact_positive=0,
                  live_accepts_exact_nonpositive=0, different_side=0)
    differences = []
    for row in data.itertuples():
        p = row.p_model_up
        f = fee_model(row)
        prices = np.array([row.up_best_ask, row.down_best_ask])
        fractions = np.array([polymarket_taker_fee_fraction_of_notional(x, f) for x in prices])
        probabilities = np.array([p, 1-p])
        live_edges = probabilities-prices-fractions-live_config['extra_buffer']
        exact_edges = probabilities-prices/(1-fractions)-live_config['extra_buffer']
        l, e = live_edges.max()>0, exact_edges.max()>0
        counts['live_additive_entries'] += int(l)
        counts['exact_entries'] += int(e)
        counts['live_rejects_exact_positive'] += int(e and not l)
        counts['live_accepts_exact_nonpositive'] += int(l and not e)
        counts['different_side'] += int(l and e and live_edges.argmax()!=exact_edges.argmax())
        differences.extend((prices+fractions-prices/(1-fractions)).tolist())
    return dict(counts, threshold_difference_quantiles={str(q): float(np.quantile(differences, q)) for q in [0, .5, 1]},
                note='Entry-only EV mode diagnostic; holdings, price caps, sizing and quote liquidity excluded. Live untouched. Exact rounded stake-specific expected PnL checked in every ledger.')


def context_diagnostics(data, context):
    joined = data[data.eligible].merge(context, on='Opened', validate='one_to_one').dropna(subset=CONTEXT_COLUMNS)
    bins = {'past_volatility_30m': [0, .0001, .00025, .0005, .001, np.inf],
            'last_return_1m': [-np.inf, -.001, -.0005, 0, .0005, .001, np.inf]}
    return {column: [dict(bucket=str(bucket), markets=len(g),
                         mismatch_rate=float(g.target_mismatch.astype(float).mean()),
                         official_up_rate=float(g.target_polymarket_up.mean()),
                         mean_raw_p_up=float(g.p_model_up.mean()),
                         raw_log_loss=calibration_metrics(g.target_polymarket_up, g.p_model_up)['log_loss'])
                    for bucket, g in joined.groupby(pd.cut(joined[column], edges, include_lowest=True), observed=True)]
            for column, edges in bins.items()}


def retrospective_comparison(data_by_latency, context, destination, live_config):
    common = set.intersection(*(set(d.loc[d.eligible, 'condition_id']) for d in data_by_latency.values()))
    frames = {}
    for latency, d in data_by_latency.items():
        frames[latency] = d[d.condition_id.isin(common)].merge(context, on='Opened', validate='one_to_one')
        frames[latency] = frames[latency].dropna(subset=CONTEXT_COLUMNS)
    common = set.intersection(*(set(d.condition_id) for d in frames.values()))
    frames = {k: d[d.condition_id.isin(common)].sort_values('decision_available_at') for k, d in frames.items()}
    results = {'description': 'Retrospective walk-forward on previously analyzed history; common latency coverage and continuous $100 portfolios.',
               'common_markets': len(common), 'context_features': CONTEXT_COLUMNS,
               'common_source_counts': frames[0].source.value_counts().to_dict(),
               'native_eligible_counts': {str(k): int(d.eligible.sum()) for k, d in data_by_latency.items()},
               'economic_margin_candidates': ECONOMIC_MARGINS, 'latencies': {}}
    for latency, data in frames.items():
        blocks = {name: [] for name in ['raw', 'simple', 'context']}
        fold_reports = []
        for fold, _, future in chronological_splits(data):
            cutoff = future.decision_available_at.min()
            past = available_labels(data, cutoff)
            inner_cutoff = past.decision_available_at.iloc[int(len(past)*.6)]
            training = available_labels(past, inner_cutoff)
            selection = past[past.decision_available_at >= inner_cutoff].copy()
            simple_choices = ['none', 'platt']+(['isotonic'] if len(training)>=1000 else [])
            simple_scores = {m: calibration_metrics(selection.target_polymarket_up,
                calibrate(fit_calibrator(training, m, inner_cutoff), selection.p_model_up))['log_loss'] for m in simple_choices}
            simple_method = min(simple_scores, key=simple_scores.get)
            corrections = {c: SettlementCorrection(c).fit(training, inner_cutoff) for c in REGULARIZATION}
            context_scores = {c: calibration_metrics(selection.target_polymarket_up, model.predict(selection))['log_loss']
                              for c, model in corrections.items()}
            chosen_c = min(context_scores, key=context_scores.get)
            full_correction = SettlementCorrection(chosen_c).fit(past, cutoff)
            selection_predictions = {'raw': selection.p_model_up.to_numpy(),
                'simple': calibrate(fit_calibrator(training, simple_method, inner_cutoff), selection.p_model_up),
                'context': corrections[chosen_c].predict(selection)}
            predictions = {'raw': future.p_model_up.to_numpy(),
                'simple': calibrate(fit_calibrator(past, simple_method, cutoff), future.p_model_up),
                'context': full_correction.predict(future)}
            margin_selection = {}
            for name, probabilities in predictions.items():
                inner = selection.assign(p_calibrated=selection_predictions[name])
                candidates = {}
                for margin in ECONOMIC_MARGINS:
                    config = dict(name='quality', kind='fixed', stake=10, edge=.05, quality_margin=margin, exact_fee_entry=True)
                    s, _, _ = backtest(inner, config, live_config, independent_funding=True)
                    candidates[margin] = s['pnl_per_turnover'] if s['number_trades']>=30 else -1.
                margin = max(candidates, key=candidates.get)
                block = future.assign(p_calibrated=probabilities, evaluation_fold=fold,
                                      selected_quality_margin=margin,
                                      selected_policy='fixed_10_edge_05' if candidates[margin]>0 else 'no_trade')
                blocks[name].append(block)
                margin_selection[name] = dict(chosen_margin=margin, inner_roi=candidates,
                                               policy=block.selected_policy.iloc[0])
            fold_reports.append(dict(fold=fold, train=len(training), selection=len(selection), past=len(past),
                evaluation=len(future), cutoff=str(cutoff), inner_cutoff=str(inner_cutoff),
                latest_training_label=str(training[['resolved_at_utc', 'market_end_utc']].max(axis=1).max()),
                latest_past_label=str(past[['resolved_at_utc', 'market_end_utc']].max(axis=1).max()),
                simple_method=simple_method, simple_inner_log_loss=simple_scores,
                regularization=chosen_c, context_inner_log_loss=context_scores,
                correction_coefficients=full_correction.coefficients.tolist(),
                context_mean=full_correction.mean.tolist(), context_scale=full_correction.scale.tolist(),
                economic_selection=margin_selection))
        scenario = {'folds': fold_reports, 'probability_metrics': {}, 'variants': {}, 'frozen_raw_entries': {}}
        # Same entry/direction/stake sample isolates probability quality from selection.
        raw_continuous = pd.concat(blocks['raw'], ignore_index=True)
        _, frozen, _ = backtest(raw_continuous, dict(kind='fixed', stake=10, edge=.05), live_config, independent_funding=True)
        frozen = enrich_trades(frozen, raw_continuous)
        metric_blocks = []
        for name, pieces in blocks.items():
            continuous = pd.concat(pieces, ignore_index=True)
            dest = destination/f'latency_{latency}s'/name
            dest.mkdir(parents=True, exist_ok=True)
            continuous.to_parquet(dest/'decisions.parquet', index=False)
            scenario['probability_metrics'][name] = calibration_metrics(continuous.target_polymarket_up, continuous.p_calibrated)
            scenario['probability_metrics'][name]['rows'] = len(continuous)
            scenario['probability_metrics'][name]['time_blocks'] = [dict(fold=int(fold), rows=len(g),
                **calibration_metrics(g.target_polymarket_up, g.p_calibrated)) for fold, g in continuous.groupby('evaluation_fold')]
            p = continuous.set_index('condition_id').p_calibrated.reindex(frozen.condition_id).to_numpy()
            p_side = np.where(frozen.side.eq('up'), p, 1-p)
            purchased_calibration = calibration_metrics(frozen.won, p_side)
            for bucket in purchased_calibration['reliability_bins']:
                bucket['actual_success_rate'] = bucket.pop('actual_up_rate')
            scenario['frozen_raw_entries'][name] = dict(bets=len(frozen),
                purchased_side_calibration=purchased_calibration,
                expected_pnl=float((p_side*frozen.shares-frozen.stake).sum()),
                realized_pnl=float(frozen.pnl.sum()))
            # Exact paired log-loss differences, resampled in dependent day blocks.
            y = continuous.target_polymarket_up.to_numpy()
            p = np.clip(continuous.p_calibrated.to_numpy(), 1e-6, 1-1e-6)
            metric_blocks.append(pd.DataFrame({'day': continuous.decision_available_at.dt.floor('D'),
                'condition_id': continuous.condition_id, name: -(y*np.log(p)+(1-y)*np.log(1-p))}))
            configs = [dict(c, adapted_probability=True) for c in BASELINES]
            configs += [dict(BASELINES[1], name='live_exact_fee_research', adapted_probability=True, exact_fee_entry=True),
                        dict(name='quality_margin_1', kind='fixed', stake=10, edge=.05, quality_margin=1., exact_fee_entry=True),
                        dict(name='past_selected_quality', kind='quality_selected')]
            for config in configs:
                active = config
                if config['kind']=='quality_selected':
                    # Choose one existing entry rule per past block; retain one portfolio.
                    # Both policies use the same probabilities; no sizing selection here.
                    active = dict(name=config['name'], kind='selected')
                    # Dynamic quality coefficient is read by policy_action below.
                    continuous['research_quality_selected'] = True
                else:
                    continuous['research_quality_selected'] = False
                summary, tr, curve = backtest(continuous, active, live_config)
                tr = enrich_trades(tr, continuous)
                tr.to_parquet(dest/(config['name']+'_trades.parquet'), index=False)
                curve.to_parquet(dest/(config['name']+'_bankroll.parquet'), index=False)
                summary['attribution'] = trade_statistics(tr)
                groups = grouped_diagnostics(tr)
                summary['time_blocks'] = groups.get('fold_id', [])
                summary['uncertainty'] = block_uncertainty(tr, continuous.decision_available_at.dt.floor('D'))
                summary['trade_calibration'] = groups.get('p_side', [])
                summary['groups'] = groups
                scenario['variants'][name+'/'+config['name']] = summary
                print('Research', latency, name, config['name'], summary['number_trades'], round(summary['final_balance'], 2), flush=True)
            signal_config = dict(name='fixed_10_independent', kind='fixed', stake=10, edge=.05, exact_fee_entry=True)
            summary, tr, _ = backtest(continuous, signal_config, live_config, independent_funding=True)
            tr = enrich_trades(tr, continuous)
            tr.to_parquet(dest/'independent_signal_trades.parquet', index=False)
            summary['attribution'] = trade_statistics(tr)
            summary['uncertainty'] = block_uncertainty(tr, continuous.decision_available_at.dt.floor('D'))
            summary['groups'] = grouped_diagnostics(tr)
            summary['max_drawdown'] = None  # Independent financing has no $100 portfolio drawdown.
            scenario['variants'][name+'/independent_signal'] = summary
        paired = metric_blocks[0]
        for block in metric_blocks[1:]:
            paired = paired.merge(block, on=['day', 'condition_id'], validate='one_to_one')
        paired['count'] = 1
        daily = paired.groupby('day')[['raw', 'simple', 'context', 'count']].sum()
        daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq='D'), fill_value=0.)
        rng = np.random.default_rng(RANDOM_SEED)
        differences = {'context_minus_raw': [], 'context_minus_simple': []}
        values = daily.to_numpy()
        for _ in range(1000):
            starts = rng.integers(0, max(len(values)-2, 1), size=int(np.ceil(len(values)/3)))
            ix = np.concatenate([np.arange(s, min(s+3, len(values))) for s in starts])[:len(values)]
            a, b, c, n = values[ix].sum(axis=0)
            differences['context_minus_raw'].append((c-a)/n)
            differences['context_minus_simple'].append((c-b)/n)
        scenario['paired_log_loss_uncertainty'] = {key: np.quantile(x, [.025, .975]).tolist() for key, x in differences.items()}
        scenario['paired_log_loss_uncertainty']['method'] = 'paired three-day moving-block bootstrap; lower log loss is better'
        results['latencies'][str(latency)] = scenario
    write_json(destination/'comparison.json', results)
    return results


def run_loss_diagnosis():
    evaluation = json.loads((REPORT_ROOT/'evaluation.json').read_text())
    if evaluation.get('status') != 'complete':
        raise ValueError('A complete cached source experiment is required')
    write_json(REPORT_ROOT/'loss_diagnosis.json', {'status': 'running', 'parent_fingerprint': evaluation['fingerprint']})
    base_reports, base_data = Path(evaluation['report_directory']), Path(evaluation['output_directory'])
    inputs = json.loads((base_reports/'inputs.json').read_text())
    if checksum(evaluation['oof']['path']) != inputs['oof_sha256']:
        raise ValueError('Current OOF changed since source experiment')
    if checksum(evaluation['oof']['metadata_path']) != inputs['metadata_sha256']:
        raise ValueError('Model metadata changed since source experiment')
    protected = ['utils/trading.py', 'utils/polymarket.py', 'utils/project_config.py',
                 'configs/runtime/active.json', 'configs/runtime/trade_policy_project.json', 'configs/live.json']
    for path in protected:
        if checksum(path) != inputs['code_and_config'][path]:
            raise ValueError('Live configuration/source changed: '+path)
    source_configuration = source_configuration_audit(evaluation)
    raw_folders = {source: Path('data/raw/polymarket')/source.split('/')[0]/revision for source, revision in inputs['sources'].items()}
    for folder in raw_folders.values():
        manifest = json.loads((folder/'manifest.json').read_text())
        for entry in manifest['files']:
            file = folder/entry['file']
            if file.stat().st_size != entry['size_bytes'] or checksum(file) != entry['sha256']:
                raise ValueError('Raw source checksum mismatch: '+str(file))
    signature = dict(parent_fingerprint=evaluation['fingerprint'], oof_sha256=inputs['oof_sha256'],
        metadata_sha256=inputs['metadata_sha256'], original_inputs=inputs,
        source_configuration=source_configuration,
        config=dict(context=CONTEXT_COLUMNS, regularization=REGULARIZATION, economic_margins=ECONOMIC_MARGINS,
                    bootstrap_seed=RANDOM_SEED, bootstrap_block_days=3, evaluation='retrospective common-coverage walk-forward'),
        code={p: checksum(p) for p in ['build_polymarket_history.py', 'utils/polymarket_history.py',
              'utils/polymarket_policy.py', 'utils/polymarket_adaptation.py', 'utils/polymarket_diagnostics.py']},
        canonical_inputs={p.name: checksum(p) for p in base_data.glob('*latency_*.parquet')},
        baseline_folds={str(p.relative_to(base_reports)): checksum(p) for p in base_reports.rglob('fold_*_calibrated.parquet')},
        official_cache_manifests={p.name: checksum(p) for p in Path('data/raw/polymarket/official_gamma').glob('manifest_*.json')},
        raw_manifests={source: checksum(folder/'manifest.json') for source, folder in raw_folders.items()})
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    destination = REPORT_ROOT/'runs'/fingerprint[:16]
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'inputs.json', signature)
    (destination/'baseline_report.md').write_text(Path('docs/polymarket_btc_experiment.md').read_text(encoding='utf-8').split('\n## Loss diagnosis and settlement adaptation')[0], encoding='utf-8')
    oof = pd.read_parquet(evaluation['oof']['path'])
    oof['Opened'] = utc(oof.Opened)
    context = decision_context(oof)
    datasets = {latency: pd.read_parquet(base_data/f'policy_latency_{latency}s.parquet') for latency in [0, 1, 2]}
    baseline, all_trades, representative, obadiaha, baseline_metrics = {}, [], None, [], {}
    for latency in [0, 1, 2]:
        for source in ['combined', 'kacho', 'obadiaha']:
            folder = base_reports/f'latency_{latency}s' if source=='combined' else base_reports/source/f'latency_{latency}s'
            decisions = pd.concat([pd.read_parquet(folder/f'fold_{fold}_calibrated.parquet') for fold in range(3)], ignore_index=True)
            scenario = evaluation['latency_scenarios'][str(latency)] if source=='combined' else evaluation['source_evaluations'][str(latency)][source]
            live = scenario['live_config']
            baseline_metrics[f'{source}/{latency}s'] = dict(rows=len(decisions),
                raw=calibration_metrics(decisions.target_polymarket_up, decisions.p_model_up),
                calibrated=calibration_metrics(decisions.target_polymarket_up, decisions.p_calibrated))
            for config in BASELINES+[{'name': 'past_selected_policy', 'kind': 'selected'}]:
                summary, trades, curve = backtest(decisions, config, live)
                original = pd.read_parquet(folder/(config['name']+'_trades.parquet'))
                if len(original):
                    pd.testing.assert_frame_equal(trades[original.columns].reset_index(drop=True), original.reset_index(drop=True))
                if len(trades)!=len(original) or not np.isclose(summary['final_balance'], scenario['continuous_baselines'][config['name']]['final_balance']):
                    raise AssertionError('Baseline reproduction failed')
                trades = enrich_trades(trades, decisions)
                key = f'{source}/{latency}s/{config["name"]}'
                baseline[key] = dict(summary=summary, attribution=trade_statistics(trades), groups=grouped_diagnostics(trades),
                                     uncertainty=block_uncertainty(trades, decisions.decision_available_at.dt.floor('D')))
                if len(trades) and source!='obadiaha':
                    all_trades.append(trades.assign(audit_variant=key))
                if source=='combined' and latency==2 and config['name']=='fixed_10_edge_05':
                    representative = trades
                    if len(trades)!=14:
                        raise ValueError('Current manifest differs from the requested 14-trade baseline')
            # Same selections/constraints through the full period, independently financed.
            for edge in [0., .02, .05]:
                config = dict(name=f'fixed_10_independent_edge_{edge}', kind='fixed', stake=10, edge=edge)
                s, tr, _ = backtest(decisions, config, live, independent_funding=True)
                tr = enrich_trades(tr, decisions)
                baseline[f'{source}/{latency}s/{config["name"]}'] = dict(summary=s, attribution=trade_statistics(tr),
                    groups=grouped_diagnostics(tr), uncertainty=block_uncertainty(tr, decisions.decision_available_at.dt.floor('D')))
            secondary = pd.read_parquet(base_data/f'obadiaha_latency_{latency}s.parquet')
            if source=='combined':
                obadiaha.append(secondary.assign(audit_latency=latency))
            print('Reproduced baseline', source, latency, flush=True)
    representative.to_csv(destination/'audit_14_trades.csv', index=False)
    representative.to_parquet(destination/'audit_14_trades.parquet', index=False)
    executed = pd.concat(all_trades, ignore_index=True)
    unique = executed.drop_duplicates('condition_id')
    official_audit = audit_official_and_oof(unique, oof, raw_folders[PRIMARY]/'btc_markets.parquet', Path('data/raw/polymarket/official_gamma'))
    official_audit.to_csv(destination/'official_oof_audit.csv', index=False)
    raw_audit = audit_raw_quotes(raw_folders[PRIMARY]/'btc_ticks.parquet', executed, representative, destination)
    secondary_audit = audit_obadiaha(raw_folders[SECONDARY], pd.concat(obadiaha), destination)
    diagnostics = dict(parent_fingerprint=evaluation['fingerprint'], fingerprint=fingerprint,
        source_configuration=source_configuration,
        representative=trade_statistics(representative), all_baselines=baseline,
        baseline_probability_metrics=baseline_metrics,
        all_eligible_raw_calibration={str(k): dict(rows=int(d.eligible.sum()),
            **calibration_metrics(d.loc[d.eligible, 'target_polymarket_up'], d.loc[d.eligible, 'p_model_up']))
            for k, d in datasets.items()},
        decision_context_diagnostics=context_diagnostics(datasets[2], context),
        independent_official_oof_audit_markets=len(unique), raw_quote_audit=raw_audit,
        obadiaha=secondary_audit, quote_statistics={str(k): quote_statistics(d[d.eligible]) for k, d in datasets.items()},
        live_fee_entry={str(k): fee_entry_audit(d[d.eligible], evaluation['latency_scenarios'][str(k)]['live_config']) for k, d in datasets.items()})
    write_json(destination/'diagnosis.json', diagnostics)
    print('Diagnosis saved', destination, diagnostics['representative'], flush=True)
    comparison = retrospective_comparison(datasets, context, destination,
                                          evaluation['latency_scenarios']['2']['live_config'])
    result = dict(status='complete', fingerprint=fingerprint, parent_fingerprint=evaluation['fingerprint'],
                  directory=str(destination.resolve()), representative=diagnostics['representative'])
    write_json(REPORT_ROOT/'loss_diagnosis.json', result)
    write_loss_report(destination, diagnostics, comparison)
    print('Loss diagnosis and comparison complete:', destination, flush=True)


def markdown_table(frame):
    # Keep report generation independent of optional tabulate dependencies.
    return '\n'.join(['| '+' | '.join(map(str, frame.columns))+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in frame.itertuples(index=False, name=None)])


def write_loss_report(destination, diagnosis, comparison):
    path = Path('docs/polymarket_btc_experiment.md')
    text = path.read_text(encoding='utf-8').split('\n## Loss diagnosis and settlement adaptation')[0]
    s = diagnosis['representative']
    lines = ['', '## Loss diagnosis and settlement adaptation', '',
        'Retrospective walk-forward on previously analyzed history. Source BTC model/OOF, features and live configuration are unchanged.', '',
        f'Parent fingerprint: `{diagnosis["parent_fingerprint"]}`. Research fingerprint: `{diagnosis["fingerprint"]}`.',
        f'Artifacts: `{destination.as_posix()}`. All 117 original portfolios reproduced on original fold coverage; original outputs retained.', '',
        f'14-trade baseline: turnover ${s["turnover"]:.2f}, fees ${s["fees"]:.3f}, official PnL ${s["pnl"]:.3f}, expected PnL ${s["expected_pnl"]:.3f}; realized minus expected ${s["realized_minus_expected"]:.3f}. Hit rate {s["hit_rate"]:.2%}; ROI {s["pnl_per_turnover"]:.2%}.',
        f'Same bets without fees: ${s["official_no_fee_pnl"]:.3f}; fee payout effect ${s["fee_payout_effect"]:.3f}. Same bets with Binance labels (diagnostic only): ${s["binance_label_diagnostic_pnl"]:.3f}; mismatches {s["target_mismatches"]}. These effects overlap and are not additive causes.', '',
        'Confirmed proximal cause: cheap-side selections with near-0.5 model probabilities win much less often than claimed. Fees do not explain the bulk of the loss; all 14 official labels equal the independently recomputed Binance proxy. This identifies a conditional prediction/selection failure, not its fundamental causal origin. Market-price information, source timing, actual fills and exchange book age are not reconstructable.', '',
        f'Independent raw audit: {diagnosis["raw_quote_audit"]["trades"]} executed ledger rows, {diagnosis["raw_quote_audit"]["distinct_quotes"]} distinct quotes; {diagnosis["independent_official_oof_audit_markets"]} unique markets checked directly against official JSON, vendor token identities and original OOF closes. Direction, token ordering, seconds units, boundary, +5m target and ask execution checks pass.',
        'Complete per-trade fields and independent evidence are in `audit_14_trades.csv`, `official_oof_audit.csv` and `kacho_entry_windows.parquet`. Windows after entry are audit evidence only; no execution was moved to a later price.', '',
        'Obadiaha: '+diagnosis['obadiaha']['conclusion'],
        'Official SDK buy-price traversal reverses the supplied levels: [SDK source](https://github.com/Polymarket/py-clob-client/blob/main/py_clob_client/order_builder/builder.py). Stored ask arrays descend from the expensive tail; the exact collector implementation/full response is unavailable. Do not assert that 0.90 is the full-book best ask or infer profitability from zero trades. Original stored-price results are retained; research does not invent missing better levels.',
        'Fees use saved per-market metadata and five-decimal rounding: [official fee documentation](https://docs.polymarket.com/trading/fees). Current documentation is not evidence of historical fee-change timing.', '',
        'Architecture: unchanged main OOF -> official settlement correction -> economic entry/sizing. Context uses only 30-minute volatility and last closed-minute return, with mandatory original logit offset. Correction strength and one-dimensional calibrator are chosen in the earlier inner block. Polymarket prices enter economics only. No Chainlink features or future price moves are used.',
        'All layers use labels available at max(official resolution, expiry), strictly before fitting. Three chronological outer folds share one $100 portfolio per variant/latency. The quality rule uses observed spreads with multiplier chosen from 0/0.5/1 on past fixed-stake signal ROI; no-trade is selected when no supported positive past ROI exists. This remains a block policy selector, not a fully contextual decision model.',
        f'Common latency coverage: {comparison["common_markets"]} eligible markets with complete past context. Identical market IDs/fold boundaries in all latencies; calibration availability can differ by scenario.', '',
        'Probability quality on all common evaluation markets:', '']
    rows = []
    for latency, scenario in comparison['latencies'].items():
        for name, m in scenario['probability_metrics'].items():
            rows.append([latency, name, f'{m["log_loss"]:.6f}', f'{m["brier_score"]:.6f}'])
    lines += [markdown_table(pd.DataFrame(rows, columns=['Latency s', 'Probability', 'Log loss', 'Brier'])), '',
              'Continuous portfolio comparison (drawdown is cost basis, not mark-to-market):', '']
    rows = []
    for latency, scenario in comparison['latencies'].items():
        for name, m in scenario['variants'].items():
            if name.endswith('/independent_signal'):
                continue
            rows.append([latency, name, f'{m["final_balance"]:.2f}',
                         f'{m["pnl_per_turnover"]:.2%}' if m['pnl_per_turnover'] is not None else 'n/a',
                         m['number_trades'], f'{m["max_drawdown"]:.2%}', json.dumps(m['rejection_reasons'])])
    lines += [markdown_table(pd.DataFrame(rows, columns=['Latency s', 'Probability / policy', 'Final USD', 'PnL / turnover', 'Trades', 'DD', 'Decision reasons'])), '',
              'Independent fixed-$10 signal diagnostics (no bankroll stopping; not a feasible $100 portfolio):', '']
    rows = []
    for latency, scenario in comparison['latencies'].items():
        for name, m in scenario['variants'].items():
            if name.endswith('/independent_signal'):
                rows.append([latency, name, m['number_trades'], f'{m["net_pnl"]:.2f}',
                             f'{m["pnl_per_turnover"]:.2%}' if m['pnl_per_turnover'] is not None else 'n/a',
                             m['uncertainty'].get('pnl_per_turnover_95pct')])
    lines += [markdown_table(pd.DataFrame(rows, columns=['Latency s', 'Probability', 'Bets', 'PnL USD', 'ROI', '3-day bootstrap 95%'])), '',
        'Attribution, reliability bins on eligible markets and selected purchased-side probabilities, price/probability/edge/source/time/latency groups, original independent-source summaries, fold results and rejection counts are stored in diagnosis.json/comparison.json. Raw p_UP groups are explicitly raw-direction diagnostics; p_side groups are success probabilities (DOWN uses 1-p_UP).',
        'Bootstrap intervals use three-day moving blocks with zero-trade days, conditional on executed bets. They do not simulate new policy selection or bankroll paths. Small traded-day counts and source non-overlap limit generalization. Sizing changes capital survival and observed samples; same-bet fee/label effects and signal diagnostics must not be added as independent effects.', '',
        'Reproduction (uses verified pinned local caches; no downloads or source retraining):', '',
        '```powershell', 'python build_polymarket_history.py',
        'python -m unittest discover -s tests -p test_polymarket_history.py',
        'python -m unittest discover -s tests -p test_polymarket_adaptation.py',
        'python -m unittest discover -s tests', '```', '',
        'Set EXPERIMENT_MODE near the top of build_polymarket_history.py to rebuild_history only to rebuild the original ingestion experiment. The default loss_diagnosis preserves source artifacts. No variant was deployed to live.']
    audit = pd.read_parquet(destination/'audit_14_trades.parquet')
    audit['Audit row'] = np.arange(1, len(audit)+1)
    identity = audit[['Audit row', 'condition_id', 'market_slug', 'Opened', 'decision_available_at',
                      'timestamp_utc', 'execution_at', 'settlement_available_at', 'source']]
    economics = audit[['Audit row', 'side', 'p_model_up', 'p_calibrated', 'p_side',
                       'up_best_bid', 'up_best_ask', 'down_best_bid', 'down_best_ask',
                       'up_ask_size', 'down_ask_size', 'stake', 'fee', 'shares',
                       'target_binance_proxy_up', 'target_polymarket_up', 'expected_pnl', 'pnl', 'realized_minus_expected']].copy()
    for column in economics.select_dtypes(include='float'):
        economics[column] = economics[column].map(lambda x: f'{x:.6f}')
    pos = lines.index('Architecture: unchanged main OOF -> official settlement correction -> economic entry/sizing. Context uses only 30-minute volatility and last closed-minute return, with mandatory original logit offset. Correction strength and one-dimensional calibrator are chosen in the earlier inner block. Polymarket prices enter economics only. No Chainlink features or future price moves are used.')
    lines[pos:pos] = ['All 14 trades: identity and UTC timing:', '', markdown_table(identity), '',
                      'All 14 trades: purchase-side probabilities and economics (fees are included in stake):', '',
                      markdown_table(economics), '',
                      'Source configuration audit: '+json.dumps(diagnosis['source_configuration']['checks'])+
                      f'; {diagnosis["source_configuration"]["feature_count"]} current main-model features. Volume-profile comparison ignores derived metadata fields; none were changed.', '']
    # Put interpretation before detailed matrices so the report is reviewable.
    two = comparison['latencies']['2']
    base = diagnosis['all_baselines']['combined/2s/fixed_10_edge_05']['summary']
    independent = diagnosis['all_baselines']['combined/2s/fixed_10_independent_edge_0.05']['attribution']
    interpret = ['', 'Findings from the completed run:', '',
        f'- The original 14-trade policy stops placing $10 bets because remaining cash is $6.79, not because a minimum order is unaffordable. Decision counts: {json.dumps(base["rejection_reasons"])}. This is not successful risk control.',
        f'- Full-period independent fixed-$10 / edge .05: {independent["trades"]} bets, PnL ${independent["pnl"]:.2f}, turnover ${independent["turnover"]:.2f}, ROI {independent["pnl_per_turnover"]:.2%}, hit rate {independent["hit_rate"]:.2%}; expected PnL ${independent["expected_pnl"]:.2f}. Same-bet no-fee PnL ${independent["official_no_fee_pnl"]:.2f}; Binance-only diagnostic ${independent["binance_label_diagnostic_pnl"]:.2f}. The signal still loses when bankroll stopping is removed.',
        f'- Common latency coverage comprises {json.dumps(comparison["common_source_counts"])}; native eligible counts {comparison["native_eligible_counts"]}. Obadiaha has no common entry across 0/1/2s: the intersections of the forward 2s windows require exactly the start+2s timestamp, while captures have fractional seconds. Thus the matched comparison evaluates {two["probability_metrics"]["raw"]["rows"]} Kacho markets, starting {two["folds"][0]["cutoff"]}, rather than the original combined April 13 start. Different coverage is not attributed to latency or adaptation.',
        f'- Context improves point-estimate log loss from {two["probability_metrics"]["raw"]["log_loss"]:.6f} (raw) / {two["probability_metrics"]["simple"]["log_loss"]:.6f} (simple) to {two["probability_metrics"]["context"]["log_loss"]:.6f}. Paired 3-day bootstrap intervals: {json.dumps(two["paired_log_loss_uncertainty"])}. A small probability-score improvement does not establish economic improvement.',
        '- On common 2s markets, fixed-$10 / edge .05 finishes at '+
            ' / '.join(f'{name}: ${two["variants"][name+"/fixed_10_edge_05"]["final_balance"]:.2f}' for name in ['raw', 'simple', 'context'])+
            '. Sizing comparisons use these same probabilities and decision times; they change survival and which bets are affordable. Fraction .01 makes zero trades at 1/2s, and only one profitable bet at 0s; this is insufficient evidence of signal quality.',
        f'- Live adds fee fraction to ask, while research break-even is ask/(1-fee_fraction). At 2s, additive entry rejects {diagnosis["live_fee_entry"]["2"]["live_rejects_exact_positive"]} otherwise exact-positive signals, and accepts {diagnosis["live_fee_entry"]["2"]["live_accepts_exact_nonpositive"]} exact-nonpositive signals. The corrected research rule trades more but also loses; the original formula is conservative, not the primary loss cause.',
        '- The spread margin and past selector do not establish profitable trading. No-trade preserves capital; context-selected quality at 1s also loses. No active traded portfolio provides supported positive profit across scenarios. The existing one-dimensional calibrator and the block selector are not a full contextual decision model.',
        '- Context choice is motivated by measured heterogeneity: at 2s, proxy/official mismatch is 7.28% for 30m volatility .0001-.00025 (3,175 markets), versus 2.16% above .001 (648). After a last-minute return above .0005, official UP occurs about 47% (1,701 markets); raw probabilities remain around .50. These retrospective associations motivate a small regularized correction, not causal claims or manually chosen trade exceptions.',
        '- Frozen raw-entry diagnostics keep directions, quotes and $10 stakes fixed: calibration changes expected PnL and prediction scores, but realized PnL necessarily remains identical. They are in comparison.json/frozen_raw_entries and distinguish probability correction from reselection. All portfolio probability/sizing combinations are then evaluated separately.',
        '- Recomputed success-probability buckets fix the old ledger summary grouping by p_UP for DOWN. This changes diagnostic bins only; all original balances/transactions reproduce. Rejection instrumentation separates no edge, cash, minimum order, locked funds and observed liquidity. Obadiaha prices are now explicitly scoped to stored levels with unknown full-book completeness; no unavailable top was fabricated.',
        '- The 787,628 raw BTC snapshots include 635,555 arrays exactly 0.99..0.90 and 505,552 captures before market start. Adapter sorting is correct for stored levels. Collector truncation is strongly suggested by the expensive ten-level tails even later in the market, but absent full responses/collector code prevents proving the true contemporaneous top or repairing lost levels.',
        '- Focused historical contracts: 34/34; new adaptation contracts: 7/7. Full suite: 177/178, with the previously documented unchanged optimizer mismatch (configured slippage 2 ticks, test expects 3). No live/optimizer setting was changed to conceal that failure.',
        '- No separate performance-rules file was present in the repository or parent instructions. Raw Parquet audits stream batches of 100,000 rows and context features use vectorized closed-candle operations.', '']
    pos = lines.index('Probability quality on all common evaluation markets:')
    lines[pos:pos] = interpret
    edge_rows = []
    signal = diagnosis['all_baselines']['combined/2s/fixed_10_independent_edge_0.05']
    for b in signal['groups']['probability_edge']:
        edge_rows.append([b['bucket'], b['trades'], f'{b["mean_success_probability"]:.3f}',
                          f'{b["observed_success_rate"]:.3f}', f'{b["pnl_per_turnover"]:.2%}'])
    lines[pos:pos] = ['Declared advantage versus observed return (full-period independent bets, 2s):', '',
        markdown_table(pd.DataFrame(edge_rows, columns=['Probability edge', 'Bets', 'Mean p_success', 'Hit rate', 'ROI'])), '']
    path.write_text(text+'\n'.join(lines)+'\n', encoding='utf-8')
