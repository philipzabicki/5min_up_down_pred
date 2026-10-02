"""Past-only market/book models, paired evaluation and execution diagnostics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.polymarket_adaptation import CONTEXT_COLUMNS, decision_context
from utils.polymarket_diagnostics import REPORT_ROOT, source_configuration_audit
from utils.polymarket_history import checksum, fee_model, payoff, write_json
from utils.polymarket_policy import calibrate, calibration_metrics, fit_calibrator

MODEL_C = (.01, .1, 1., 10.)
FIXED_STAKE = 5.
BOOTSTRAP_SEED = 20261003
BOOTSTRAP_BLOCK_DAYS = 3
EXECUTION_DELAYS = (0, 1, 2)
OOF_LATENCIES = (0, 1, 2)
MARKET_COLUMNS = [
    'up_best_bid', 'up_best_ask', 'down_best_bid', 'down_best_ask',
    'up_mid', 'down_mid', 'up_spread', 'down_spread', 'sum_bids', 'sum_asks',
    'p_market_mid', 'log_ask_size_up', 'log_ask_size_down',
    'log_bid_size_up', 'log_bid_size_down', 'ask_size_imbalance', 'bid_size_imbalance',
]
MODEL_COLUMNS = {
    'market_only': MARKET_COLUMNS,
    'market_plus_oof': MARKET_COLUMNS+['source_logit'],
    'market_plus_oof_context': MARKET_COLUMNS+['source_logit']+CONTEXT_COLUMNS,
}
MODEL_VARIANTS = ('market_mid', 'oof_raw', 'oof_platt', *MODEL_COLUMNS)


def market_features(data):
    """Transform only the contemporaneous, confirmed Kacho quote; quantities are shares."""
    x = pd.DataFrame(index=data.index)
    for c in ['up_best_bid', 'up_best_ask', 'down_best_bid', 'down_best_ask']:
        x[c] = data[c].astype(float)
    for side in ['up', 'down']:
        x[side+'_mid'] = (data[side+'_best_bid']+data[side+'_best_ask'])/2
        x[side+'_spread'] = data[side+'_best_ask']-data[side+'_best_bid']
    x['sum_bids'] = data.up_best_bid+data.down_best_bid
    x['sum_asks'] = data.up_best_ask+data.down_best_ask
    x['p_market_mid'] = x.up_mid/(x.up_mid+x.down_mid)
    for side in ['up', 'down']:
        x['log_ask_size_'+side] = np.log1p(data[side+'_ask_size'].clip(lower=0))
        x['log_bid_size_'+side] = np.log1p(data[side+'_bid_size'].clip(lower=0))
    x['ask_size_imbalance'] = ((data.up_ask_size-data.down_ask_size)/
                               (data.up_ask_size+data.down_ask_size).replace(0, np.nan))
    x['bid_size_imbalance'] = ((data.up_bid_size-data.down_bid_size)/
                               (data.up_bid_size+data.down_bid_size).replace(0, np.nan))
    return x[MARKET_COLUMNS]


def prepare_common_data(root, output, oof_path):
    """Use Kacho only and hold market IDs fixed across source and quote latencies."""
    frames = {}
    coverage_by_latency = {}
    for latency in OOF_LATENCIES:
        path = root/f'kacho_latency_{latency}s.parquet'
        d = pd.read_parquet(path)
        sizes = d[['up_bid_size', 'down_bid_size', 'up_ask_size', 'down_ask_size']]
        complete_sizes = sizes.notna().all(axis=1) & np.isfinite(sizes).all(axis=1) & sizes.ge(0).all(axis=1)
        valid = d.eligible & d.quote_valid.fillna(False) & complete_sizes
        coverage_by_latency[latency] = dict(
            eligible=int(d.eligible.sum()),
            eligible_with_valid_quote_and_all_sizes=int(valid.sum()))
        frames[latency] = d.loc[valid].copy()
    ids = set.intersection(*(set(d.condition_id) for d in frames.values()))
    if not ids:
        raise ValueError('No complete Kacho market-book coverage common to all source latencies')
    frames = {latency: d[d.condition_id.isin(ids)].copy().sort_values('market_start_utc')
              for latency, d in frames.items()}
    reference = frames[2]
    if reference.condition_id.duplicated().any():
        raise ValueError('Common Kacho market IDs are not unique')
    context = decision_context(pd.read_parquet(oof_path))
    for latency, d in frames.items():
        d = d.merge(context, on='Opened', validate='one_to_one')
        if (d.timestamp_utc < d.decision_available_at).any() or (
                d.timestamp_utc > d.decision_available_at+pd.Timedelta(seconds=2)).any():
            raise ValueError('Market-book features precede availability or exceed the quote tolerance')
        if (d.context_available_at > d.decision_available_at).any():
            raise ValueError('Source context was not available when its book was observed')
        features = market_features(d)
        if not np.isfinite(features.to_numpy()).all():
            raise ValueError('Invalid or missing market-book feature in common sample')
        d['p_market_mid'] = features.p_market_mid
        d['market_book_observed_at'] = d.timestamp_utc
        d['second_layer_computed_at'] = d.timestamp_utc
        d['earliest_execution_at'] = d.timestamp_utc
        d['source_latency_s'] = latency
        frames[latency] = d
        d.to_parquet(output/f'common_latency_{latency}s.parquet', index=False)
        coverage_by_latency[latency]['common_markets'] = len(d)
        coverage_by_latency[latency]['removed_for_incomplete_quote_or_sizes'] = (
            coverage_by_latency[latency]['eligible']-
            coverage_by_latency[latency]['eligible_with_valid_quote_and_all_sizes'])
        coverage_by_latency[latency]['valid_but_outside_common_intersection'] = (
            coverage_by_latency[latency]['eligible_with_valid_quote_and_all_sizes']-len(d))
    if any(not frames[0].condition_id.reset_index(drop=True).equals(frames[x].condition_id.reset_index(drop=True))
           for x in (1, 2)):
        raise AssertionError('Source-latency market coverage is not paired')
    return frames, dict(common_markets=len(ids), source='kachoio/polymarket-5-minute-crypto-up-down-markets',
        by_latency={str(k):v for k,v in coverage_by_latency.items()},
        observed_at='First forward Kacho tick at/after source OOF availability (latency 0/1/2s); max quote delay 2s.',
        second_layer='Computed at the contemporaneous observed tick; zero extra compute delay assumption.',
        quantities='Best bid/ask sizes are shares. Price and all four best-level quantities must be finite; negative quantities are excluded.',
        book_age='Exchange-side book age is unknown; recording cadence is not used as an age proxy.')


def _matrix(data, variant):
    market = market_features(data)
    market['source_logit'] = logit(data.p_model_up.clip(1e-6, 1-1e-6))
    x = market.join(data[CONTEXT_COLUMNS])
    return x[MODEL_COLUMNS[variant]]


def fit_market_model(train, variant, strength, label_cutoff):
    if variant not in MODEL_COLUMNS:
        raise ValueError('Unknown market model variant')
    if not (train[['resolved_at_utc', 'market_end_utc']].max(axis=1) < label_cutoff).all():
        raise ValueError('Model training data includes an unavailable official label')
    model = make_pipeline(StandardScaler(), LogisticRegression(C=strength, max_iter=1000, solver='lbfgs'))
    model.fit(_matrix(train, variant), train.target_polymarket_up.astype(int))
    return model


def score_probabilities(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1-1e-8)
    y = np.asarray(y, dtype=int)
    return {'rows': len(y), 'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'brier': float(brier_score_loss(y, p)),
            'auc': float(roc_auc_score(y, p)) if np.unique(y).size==2 else None,
            'calibration_bins': calibration_metrics(y, p)['reliability_bins']}


def paired_block_bootstrap(data, left, right):
    """Same market IDs in each side and every draw; 3-day moving calendar blocks."""
    x = data[['market_start_utc', 'target_polymarket_up']].copy()
    y = x.target_polymarket_up.to_numpy()
    out = {}
    for metric in ['log_loss', 'brier']:
        a, b = np.clip(data[left].to_numpy(), 1e-8, 1-1e-8), np.clip(data[right].to_numpy(), 1e-8, 1-1e-8)
        if metric=='log_loss':
            xa = -(y*np.log(a)+(1-y)*np.log(1-a))
            xb = -(y*np.log(b)+(1-y)*np.log(1-b))
        else:
            xa, xb = (y-a)**2, (y-b)**2
        daily = pd.DataFrame({'day': pd.to_datetime(x.market_start_utc, utc=True).dt.floor('D'),
                              'a': xa, 'b': xb, 'n': 1}).groupby('day').sum()
        days = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
        v = daily.reindex(days, fill_value=0).to_numpy()
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        diffs = []
        for _ in range(2000):
            starts = rng.integers(0, max(len(v)-BOOTSTRAP_BLOCK_DAYS+1, 1),
                                 size=int(np.ceil(len(v)/BOOTSTRAP_BLOCK_DAYS)))
            ix = np.concatenate([np.arange(s, min(s+BOOTSTRAP_BLOCK_DAYS, len(v))) for s in starts])[:len(v)]
            av, bv, n = v[ix].sum(axis=0)
            if n:
                diffs.append((av-bv)/n)
        out[metric+'_left_minus_right_95pct'] = np.quantile(diffs, [.025, .975]).tolist()
    out.update(left=left, right=right, markets=len(data), calendar_days=len(days),
               block_days=BOOTSTRAP_BLOCK_DAYS, replications=2000,
               note='Paired by market and sampled in identical 3-calendar-day blocks; negative favors left.')
    return out


def walk_forward_models(frames, destination):
    results = {'outer_folds': 3, 'warmup_fraction': .4, 'model_C_candidates': MODEL_C,
               'same_market_partition_by_source_latency': True, 'latencies': {}}
    all_models = tuple(MODEL_VARIANTS)
    for latency, data in frames.items():
        data = data.sort_values('market_start_utc').reset_index(drop=True)
        unique = data.market_start_utc.drop_duplicates().reset_index(drop=True)
        boundaries = [int(len(unique)*fraction) for fraction in (.4, .6, .8)]+[len(unique)]
        if len(set(boundaries)) != len(boundaries):
            raise ValueError('Insufficient distinct markets for three chronological folds')
        fold_rows, fold_reports = [], []
        for fold in range(3):
            start = unique.iloc[boundaries[fold]]
            end = unique.iloc[boundaries[fold+1]] if fold < 2 else None
            outer_time = data.loc[data.market_start_utc.ge(start), 'decision_available_at'].min()
            in_fold = data.market_start_utc.ge(start)
            if end is not None:
                in_fold &= data.market_start_utc.lt(end)
            future = data[in_fold].copy()
            past = data[(data.market_start_utc < start) &
                        (data[['resolved_at_utc', 'market_end_utc']].max(axis=1) < outer_time)].copy()
            if len(past) < 500 or len(future) < 100:
                raise ValueError(f'Insufficient past/future observations for outer fold {fold}')
            inner = past.market_start_utc.iloc[int(len(past)*.6)]
            train = past[(past.market_start_utc < inner) &
                         (past[['resolved_at_utc', 'market_end_utc']].max(axis=1) < past.loc[past.market_start_utc.ge(inner), 'decision_available_at'].min())].copy()
            selection = past[past.market_start_utc.ge(inner) &
                (past[['resolved_at_utc','market_end_utc']].max(axis=1)<outer_time)].copy()
            if len(train)<300 or len(selection)<100 or not (
                    train[['resolved_at_utc','market_end_utc']].max(axis=1).max() < selection.decision_available_at.min()):
                raise ValueError('Inner model training labels are not strictly available before selection')
            chosen, inner_scores = {}, {}
            for variant in MODEL_COLUMNS:
                scores = {}
                for strength in MODEL_C:
                    model = fit_market_model(train, variant, strength, selection.decision_available_at.min())
                    scores[str(strength)] = float(log_loss(selection.target_polymarket_up,
                                                            model.predict_proba(_matrix(selection, variant))[:, 1], labels=[0,1]))
                best = min(MODEL_C, key=lambda value: scores[str(value)])
                chosen[variant], inner_scores[variant] = best, scores
            chosen['oof_platt'], chosen['oof_raw'], chosen['market_mid'] = None, None, None
            predicted = {}
            predicted['oof_raw'] = future.p_model_up.to_numpy()
            available = past[['resolved_at_utc','market_end_utc']].max(axis=1)
            if not (available < outer_time).all():
                raise ValueError('OOF calibration includes an unavailable official label')
            predicted['oof_platt'] = calibrate(fit_calibrator(past, 'platt', outer_time), future.p_model_up)
            predicted['market_mid'] = future.p_market_mid.to_numpy()
            for variant in MODEL_COLUMNS:
                model = fit_market_model(past, variant, chosen[variant], outer_time)
                predicted[variant] = model.predict_proba(_matrix(future, variant))[:, 1]
            block = future[['condition_id','market_slug','market_start_utc','decision_available_at','timestamp_utc',
                'market_end_utc','resolved_at_utc','target_polymarket_up','p_model_up','source_latency_s',
                'up_best_bid','up_best_ask','down_best_bid','down_best_ask','up_bid_size','down_bid_size',
                'up_ask_size','down_ask_size','fee_rate','fee_exponent','fee_round_decimals','fee_min_fee',
                'order_min_size','tick_size','quote_delay_ms','seconds_to_expiry']].copy()
            block['fold'] = fold
            for name, values in predicted.items():
                block[name] = values
            fold_rows.append(block)
            fold_reports.append(dict(fold=fold, train_rows=len(train), selection_rows=len(selection),
                past_rows=len(past), evaluation_rows=len(future),
                evaluation_start=str(future.market_start_utc.min()), evaluation_end=str(future.market_start_utc.max()),
                latest_training_label_available=str(train[['resolved_at_utc','market_end_utc']].max(axis=1).max()),
                selection_starts=str(selection.market_start_utc.min()), cutoff=str(outer_time),
                selected_C={k:(float(v) if v is not None else None) for k,v in chosen.items()},
                inner_log_loss_by_C=inner_scores))
        oof = pd.concat(fold_rows, ignore_index=True)
        out = destination/f'latency_{latency}s'
        out.mkdir(parents=True, exist_ok=True)
        oof.to_parquet(out/'out_of_fold_probabilities.parquet', index=False)
        metrics = {name: score_probabilities(oof.target_polymarket_up, oof[name]) for name in all_models}
        metrics['folds'] = {str(fold): {name: score_probabilities(g.target_polymarket_up, g[name])
                          for name in all_models} for fold,g in oof.groupby('fold')}
        metrics['paired_uncertainty'] = {
            'market_plus_oof_minus_market_only': paired_block_bootstrap(oof,'market_plus_oof','market_only'),
            'context_minus_market_plus_oof': paired_block_bootstrap(oof,'market_plus_oof_context','market_plus_oof'),
        }
        raw_subgroups = []
        compare = oof.assign(pmarket=oof.market_mid, p_source=oof.p_model_up,
                              source_market_gap=(oof.p_model_up-oof.market_mid).abs(),
                              chosen_side=np.where(oof.p_model_up.ge(.5),'up','down'))
        for name, values in [
            ('entry_ask', pd.cut(compare[['up_best_ask','down_best_ask']].min(axis=1),
                                 [0,.3,.4,.5,.6,1],include_lowest=True)),
            ('maximum_spread',pd.cut(pd.concat([compare.up_best_ask-compare.up_best_bid,
                                                 compare.down_best_ask-compare.down_best_bid],axis=1).max(axis=1),
                                     [0,.01,.02,.05,.1,1],include_lowest=True)),
            ('source_market_probability_gap',pd.cut(compare.source_market_gap,[0,.01,.025,.05,.1,.2,.5],include_lowest=True)),
            ('source_predicted_direction',compare.chosen_side)]:
            raw_subgroups.append({'dimension':name,'bins':[dict(bucket=str(bucket), markets=len(g),
                **{v:score_probabilities(g.target_polymarket_up,g[v]) for v in ['market_only','market_plus_oof']},
                oof_minus_market_log_loss=score_probabilities(g.target_polymarket_up,g.market_plus_oof)['log_loss']-
                    score_probabilities(g.target_polymarket_up,g.market_only)['log_loss'])
                for bucket,g in compare.groupby(values,observed=True) if len(g)>=50]})
        metrics['diagnostic_subgroups'] = raw_subgroups
        metrics['point_difference_market_plus_oof_minus_market_only'] = {
            k: metrics['market_plus_oof'][k]-metrics['market_only'][k] for k in ['log_loss','brier','auc']}
        metrics['point_difference_context_minus_market_plus_oof'] = {
            k: metrics['market_plus_oof_context'][k]-metrics['market_plus_oof'][k] for k in ['log_loss','brier','auc']}
        metrics['fold_reports'] = fold_reports
        metrics['rows'] = len(oof)
        metrics['evaluation_period'] = [str(oof.market_start_utc.min()),str(oof.market_start_utc.max())]
        metrics['inner_tuning_budget'] = {'family':'StandardScaler + L2 LogisticRegression','C':MODEL_C,
            'same_for_market_only_and_market_plus_oof':True,'outer_test_used_for_tuning':False,
            'early_stopping':'Not used in second layer.'}
        write_json(out/'scores.json',metrics)
        results['latencies'][str(latency)] = metrics
        print('Market value scored',latency,len(oof),metrics['point_difference_market_plus_oof_minus_market_only'],flush=True)
    return results


def decide_at_observed_book(row, probability):
    """Select side and immutable ask limit using only the book at observation time."""
    choices=[]
    positive_but_unfilled=[]
    fees=fee_model(row)
    for side, p in [('up', probability), ('down', 1-probability)]:
        price=getattr(row,side+'_best_ask')
        size=getattr(row,side+'_ask_size')
        result=payoff(FIXED_STAKE,price,1,fees,np.inf)
        if not result:
            continue
        ev=p*result['shares']-FIXED_STAKE
        if ev<=0:
            continue
        if result['shares']<row.order_min_size:
            positive_but_unfilled.append('observed_minimum_order')
            continue
        if size<result['shares']:
            positive_but_unfilled.append('observed_liquidity')
            continue
        if ev>0:
            choices.append((ev,side,price,result,p))
    return (max(choices,key=lambda x:x[0]),[]) if choices else (None,
        positive_but_unfilled if positive_but_unfilled else ['no_positive_expected_pnl'])


def _execution_quotes(data, quote_path, delay):
    from utils.polymarket_history import select_quotes
    order=data[['decision_id','condition_id','market_start_utc','market_end_utc','timestamp_utc']].copy()
    order['observed_book_at']=order.timestamp_utc
    order['decision_available_at']=order.timestamp_utc+pd.Timedelta(seconds=delay)
    future=select_quotes(order.drop(columns='timestamp_utc'),quote_path,2000)
    # Keep future data strictly in the execution envelope; zero delay is the observed quote.
    future=future.rename(columns={'timestamp_utc':'execution_quote_at',
        'up_best_bid':'up_best_bid_quote','up_best_ask':'up_best_ask_quote',
        'down_best_bid':'down_best_bid_quote','down_best_ask':'down_best_ask_quote',
        'up_bid_size':'up_bid_size_quote','down_bid_size':'down_bid_size_quote',
        'up_ask_size':'up_ask_size_quote','down_ask_size':'down_ask_size_quote'})
    return future


def simulate_fixed_policy(data, probability_column, execution_delay=0, bankroll=None, future_quotes=None):
    """Freeze side/limit at observation; recheck later ask, capacity, cash and settlement."""
    ordered=data.sort_values('market_start_utc').reset_index(drop=True)
    cash=bankroll
    locked=0.
    pending=[]
    trades=[]
    reasons={}
    curve=[]
    equity_peak=bankroll
    max_dd=0.

    def reject(reason):
        reasons[reason]=reasons.get(reason,0)+1

    def settle(until):
        nonlocal cash,locked,max_dd,equity_peak
        while pending and pending[0][0]<=until:
            when,stake,payout=__import__('heapq').heappop(pending)
            cash+=payout; locked-=stake
            if not pending: locked=0.
            if cash is not None:
                equity=cash+locked
                equity_peak=max(equity_peak,equity)
                max_dd=max(max_dd,1-equity/equity_peak)
            curve.append(dict(timestamp_utc=when,event='settlement',cash=cash,open_cost=locked,
                              cost_basis_equity=cash+locked))

    for row in ordered.itertuples():
        observed,observed_rejections=decide_at_observed_book(row,getattr(row,probability_column))
        if observed is None:
            for reason in observed_rejections:
                reject(reason)
            continue
        ev,side,limit,obs_pay,p_success=observed
        if future_quotes is None:
            quote=row
            execution_at=row.timestamp_utc
        else:
            quote=future_quotes.loc[row.condition_id]
            execution_at=quote.execution_quote_at
            if pd.isna(execution_at) or execution_at<row.timestamp_utc+pd.Timedelta(seconds=execution_delay) or execution_at>=row.market_end_utc:
                reject('no_forward_execution_quote'); continue
            if not bool(quote.quote_valid):
                reject('invalid_execution_quote'); continue
        settle(execution_at)
        price=getattr(quote,side+'_best_ask'+('_quote' if future_quotes is not None else ''))
        size=getattr(quote,side+'_ask_size'+('_quote' if future_quotes is not None else ''))
        if price>limit+1e-12:
            reject('execution_price_above_observed_limit'); continue
        fees=fee_model(row)
        execution=payoff(FIXED_STAKE,price,1,fees,size)
        if execution is None:
            reject('insufficient_execution_ask_liquidity'); continue
        if execution['shares']<row.order_min_size:
            reject('minimum_order_at_execution'); continue
        if cash is not None and cash < FIXED_STAKE-1e-9:
            reject('insufficient_cash'); continue
        outcome=row.target_polymarket_up if side=='up' else 1-row.target_polymarket_up
        pnl=execution['shares']*outcome-FIXED_STAKE
        trades.append(dict(condition_id=row.condition_id,market_slug=row.market_slug,fold=row.fold,
            source_latency_s=row.source_latency_s,observed_at=row.timestamp_utc,
            oof_available_at=row.decision_available_at,second_layer_computed_at=row.second_layer_computed_at,
            execution_at=execution_at,execution_quote_at=execution_at,side=side,p_success=p_success,
            p_model_up=row.p_model_up,p_market_mid=row.p_market_mid,observed_limit=limit,execution_price=price,
            observed_expected_pnl=ev,execution_expected_pnl=p_success*execution['shares']-FIXED_STAKE,
            stake=FIXED_STAKE,fee=execution['fee'],shares=execution['shares'],
            target_polymarket_up=row.target_polymarket_up,pnl=pnl,entry_return=pnl/FIXED_STAKE,
            observed_ask_size=getattr(row,side+'_ask_size'),execution_ask_size=size,
            seconds_to_expiry=(row.market_end_utc-execution_at).total_seconds()))
        if cash is not None:
            cash-=FIXED_STAKE; locked+=FIXED_STAKE
            import heapq
            heapq.heappush(pending,(max(row.resolved_at_utc,row.market_end_utc),FIXED_STAKE,FIXED_STAKE+pnl))
            equity=cash+locked
            equity_peak=max(equity_peak,equity)
            max_dd=max(max_dd,1-equity/equity_peak)
        reasons['executed']=reasons.get('executed',0)+1
        curve.append(dict(timestamp_utc=execution_at,event='decision',cash=cash,open_cost=locked,
                          cost_basis_equity=None if cash is None else cash+locked))
    if cash is not None:
        settle(pd.Timestamp.max.tz_localize('UTC'))
    trades=pd.DataFrame(trades)
    turnover=float(trades.stake.sum()) if len(trades) else 0.
    fold_economics={}
    for fold in sorted(ordered.fold.dropna().unique()):
        block=trades[trades.fold.eq(fold)] if len(trades) else pd.DataFrame()
        block_turnover=float(block.stake.sum()) if len(block) else 0.
        fold_economics[str(int(fold))]=dict(trades=len(block),turnover=block_turnover,
            fees=float(block.fee.sum()) if len(block) else 0.,
            expected_pnl=float(block.execution_expected_pnl.sum()) if len(block) else 0.,
            realized_pnl=float(block.pnl.sum()) if len(block) else 0.,
            pnl_per_turnover=float(block.pnl.sum()/block_turnover) if block_turnover else None,
            hit_rate=float(block.pnl.gt(0).mean()) if len(block) else None)
    summary=dict(trades=len(trades),turnover=turnover,fees=float(trades.fee.sum()) if len(trades) else 0.,
        expected_pnl=float((trades.execution_expected_pnl).sum()) if len(trades) else 0.,
        realized_pnl=float(trades.pnl.sum()) if len(trades) else 0.,
        pnl_per_turnover=float(trades.pnl.sum()/turnover) if turnover else None,
        hit_rate=float(trades.pnl.gt(0).mean()) if len(trades) else None,rejection_reasons=reasons,
        by_evaluation_fold=fold_economics,
        independent_funding=bankroll is None,execution_delay_s=execution_delay)
    summary.update(initial_bankroll=bankroll,final_balance=cash,max_drawdown=None if cash is None else max_dd,
                   drawdown_basis='cash + locked cost; not mark-to-market')
    return summary,trades,pd.DataFrame(curve)


def evaluate_execution(frames, scores, quote_path, destination):
    comparisons={}
    prediction_columns=['market_mid','oof_raw','oof_platt',*MODEL_COLUMNS]
    for latency,data in frames.items():
        scored=scores['latencies'][str(latency)]
        oof=pd.read_parquet(destination/f'latency_{latency}s'/'out_of_fold_probabilities.parquet')
        data=data[data.condition_id.isin(oof.condition_id)].merge(oof[['condition_id','fold',*prediction_columns]],
            on='condition_id',validate='one_to_one')
        by_latency={}
        for delay in EXECUTION_DELAYS:
            future=None if delay==0 else _execution_quotes(data,quote_path,delay).set_index('condition_id')
            variant_results={}
            for name in prediction_columns:
                independent,ledger,_=simulate_fixed_policy(data,name,delay,None,future)
                portfolio,portfolio_ledger,path=simulate_fixed_policy(data,name,delay,100.,future)
                if not np.isclose(portfolio['final_balance'],100+portfolio['realized_pnl']):
                    raise AssertionError('Fixed-stake portfolio cash flow does not reconcile')
                ledger.to_parquet(destination/f'latency_{latency}s'/f'execution_{delay}s_{name}_independent.parquet',index=False)
                portfolio_ledger.to_parquet(destination/f'latency_{latency}s'/f'execution_{delay}s_{name}_portfolio.parquet',index=False)
                path.to_parquet(destination/f'latency_{latency}s'/f'execution_{delay}s_{name}_path.parquet',index=False)
                variant_results[name]=dict(independent=independent,portfolio=portfolio)
            by_latency[str(delay)]=variant_results
        comparisons[str(latency)]=by_latency
        print('Market value economics',latency,flush=True)
    return comparisons


def run_market_value_experiment():
    base=REPORT_ROOT/'evaluation.json'
    evaluation=json.loads(base.read_text())
    if evaluation.get('status')!='complete':
        raise ValueError('The current source evaluation manifest is incomplete')
    parent=Path(evaluation['output_directory'])
    reports=Path(evaluation['report_directory'])
    original=json.loads((reports/'inputs.json').read_text())
    if checksum(evaluation['oof']['path'])!=original['oof_sha256']:
        raise ValueError('The current source OOF checksum differs from its manifest')
    if checksum(evaluation['oof']['metadata_path'])!=original['metadata_sha256']:
        raise ValueError('The model metadata changed since the source run')
    config=source_configuration_audit(evaluation)
    protected=['utils/trading.py','utils/polymarket.py','utils/project_config.py',
        'configs/runtime/active.json','configs/runtime/trade_policy_project.json','configs/live.json']
    for path in protected:
        if checksum(path)!=original['code_and_config'][path]:
            raise ValueError('Live configuration/source changed since the parent evaluation: '+path)
    raw_manifest=Path('data/raw/polymarket/kachoio')/original['sources']['kachoio/polymarket-5-minute-crypto-up-down-markets']/'manifest.json'
    raw=json.loads(raw_manifest.read_text())
    raw_checks=[]
    for file in raw['files']:
        path=raw_manifest.parent/file['file']
        raw_checks.append(path.stat().st_size==file['size_bytes'] and checksum(path)==file['sha256'])
    if not all(raw_checks):
        raise ValueError('A revision-pinned Kacho source checksum failed')
    signature=dict(parent_fingerprint=evaluation['fingerprint'],oof_sha256=original['oof_sha256'],
        model_metadata_sha256=original['metadata_sha256'],kacho_manifest_sha256=checksum(raw_manifest),
        canonical_inputs={str(parent/f'kacho_latency_{k}s.parquet'):checksum(parent/f'kacho_latency_{k}s.parquet') for k in OOF_LATENCIES},
        market_data={str(parent/'quotes.parquet'):checksum(parent/'quotes.parquet')},
        config=config,
        protected_live_code_and_config={p:checksum(p) for p in protected},
        time_contract={'oof_latencies':OOF_LATENCIES,'max_quote_delay_s':2,'execution_delays_s':EXECUTION_DELAYS,
           'quote_selection':'first forward recorded quote; no nearest/interpolation/carry; strict before expiry',
           'model_observation':'same quote as features, first forward quote after each source-availability scenario',
           'computed_at':'book observation timestamp, zero additional compute time assumed',
           'execution_limit':'observed selected-side ask; never reselect after later quote'},
        model={'family':'StandardScaler + L2 logistic regression','C_candidates':MODEL_C,'stake':FIXED_STAKE,
               'outer':'three chronological walk-forward folds, first 40% warmup',
               'inner':'past-only chronological validation and official-label availability gate'},
        code={p:checksum(p) for p in ['utils/polymarket_market_value.py','utils/polymarket_adaptation.py',
             'utils/polymarket_diagnostics.py','utils/polymarket_history.py','utils/polymarket_policy.py',
             'build_polymarket_history.py']})
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    destination=REPORT_ROOT/'runs'/fingerprint[:16]
    destination.mkdir(parents=True,exist_ok=True)
    write_json(destination/'inputs.json',signature)
    frames,coverage=prepare_common_data(parent,destination,evaluation['oof']['path'])
    scores=walk_forward_models(frames,destination)
    economics=evaluate_execution(frames,scores,parent/'quotes.parquet',destination)
    result={'status':'complete','fingerprint':fingerprint,'parent_fingerprint':evaluation['fingerprint'],
            'directory':str(destination.resolve()),'coverage':coverage,'scores':scores,'economics':economics,
            'obadiaha_excluded':'Full-book top level is unverified; stored asks are not treated as executable prices.',
            'limitations':['Retrospective walk-forward on previously analyzed period, not a pristine holdout.',
                'Kacho records a cached book at 1Hz; actual CLOB book age, fill and quote persistence are unknown.',
                'Zero additional second-layer calculation delay is optimistic; 1/2s use first observed forward quote and fixed decision limit.',
                'Historical Gamma fee schedule was fetched later; fee changes cannot be dated.',
                'Existing OOF construction and its accepted evaluation-fold early stopping are unchanged.',
                'No other policies, slippage or settlement timing are changed.']}
    write_json(destination/'market_value.json',result)
    write_json(REPORT_ROOT/'market_value.json',{k:v for k,v in result.items() if k not in ['scores','economics']})
    append_market_value_report(destination,result)
    print('Market-value experiment complete',fingerprint,destination,flush=True)
    return result


def append_market_value_report(destination,result):
    path=Path('docs/polymarket_btc_experiment.md')
    text=path.read_text(encoding='utf-8').split('\n## Incremental Polymarket information value')[0]
    lines=['','## Incremental Polymarket information value','','Question: does the unchanged BTC OOF improve official settlement prediction and after-cost returns beyond the Kacho book? This is a retrospective walk-forward on previously analyzed history.',
       '',f'Fingerprint: `{result["fingerprint"]}`; parent source experiment: `{result["parent_fingerprint"]}`. Full models, paired bootstrap, calibration bins, fold scores, fixed-$5 trade ledgers and portfolio paths: `{destination.as_posix()}`.',
       f'Common sample: {result["coverage"]["common_markets"]:,} eligible Kacho markets; {result["scores"]["latencies"]["0"]["rows"]:,} markets per latency receive outer-fold predictions after chronological warmup. Obadiaha excluded because stored levels do not confirm full-book best prices. At each source latency, quote features use the first recorded Kacho tick at or after prediction availability, with no lookahead; quote/book observation, second-layer calculation, and zero-delay execution start at that tick. Additional execution quotes are first observations at +1s/+2s. Direction and ask limit stay fixed from the observation.',
       'Market-only and market-plus-OOF models use identical L2 logistic families, C grid, inner/outer market partitions, and label availability rules. Scaling fits the training data within each fold. Market inputs: four best prices, four best-level quantities (shares), mids, spreads, bid/ask sums, normalized mid estimator, log sizes and bid/ask size imbalances. Bid aggregate USD depth is not substituted for best-level shares. No future quotes, outcomes, IDs or target disagreement enter features.',
       'OOF variants report raw OOF and the existing Platt calibration. MARKET_ONLY, MARKET_PLUS_OOF and MARKET_PLUS_OOF_CONTEXT are the primary comparisons. `market_mid` is the normalized-mid benchmark, not a guaranteed fair price.','','Prediction quality on identical markets within each source-availability scenario:','']
    rows=[]
    for latency,met in result['scores']['latencies'].items():
        for name in MODEL_VARIANTS:
            m=met[name]
            rows.append([latency,name,m['rows'],f'{m["log_loss"]:.6f}',f'{m["brier"]:.6f}',
                         f'{m["auc"]:.4f}' if m['auc'] is not None else 'n/a'])
    lines+=_mdtable(['OOF availability s','Variant','Markets','Log loss','Brier','AUC'],rows)+['',
        'Paired differences, left minus right; negative favours the model added on the left. The same three-calendar-day blocks and market IDs are resampled in 2,000 replicates:','']
    rows=[]
    for latency,met in result['scores']['latencies'].items():
        for key in ['market_plus_oof_minus_market_only','context_minus_market_plus_oof']:
            b=met['paired_uncertainty'][key]
            rows.append([latency,b['left'],b['right'],json.dumps(b['log_loss_left_minus_right_95pct']),
                         json.dumps(b['brier_left_minus_right_95pct']),b['markets'],b['block_days']])
    lines+=_mdtable(['OOF availability s','Left','Right','Log-loss delta 95%','Brier delta 95%','Markets','Block days'],rows)+['',
        'Interpretation: at 0s, the paired Brier interval for market_plus_oof minus market_only excludes zero and favors adding OOF, while the log-loss interval includes zero. At 1s and 2s, both metric intervals include zero. The small 0s Brier improvement does not establish profitability or rule out predictive value on the Binance target. This sample does not establish stable incremental settlement-prediction value beyond the contemporaneous Kacho book. Raw OOF alone also scores worse than the normalized-mid and market-only baselines here. This is retrospective evidence, not a prospective guarantee.','',
        'Inner-fold log-loss selection and outer-fold scores are stored fold by fold. Fixed probability calibration bins and market price/spread/direction/source disagreement subgroup tables are diagnostic and do not determine a new rule. AUC is secondary.','','Fixed $5 economics; these compare the same entry calculation across probabilities. Independent funding has no $100 balance or drawdown:','']
    rows=[]
    for latency,delays in result['economics'].items():
        for delay,variants in delays.items():
            for name,vals in variants.items():
                a,b=vals['independent'],vals['portfolio']
                rows.append([latency,delay,name,a['trades'],f'{a["realized_pnl"]:.2f}',
                    f'{a["expected_pnl"]:.2f}',f'{a["pnl_per_turnover"]:.2%}' if a['pnl_per_turnover'] is not None else 'n/a',
                    b['trades'],f'{b["final_balance"]:.2f}',f'{b["max_drawdown"]:.2%}',json.dumps(a['rejection_reasons'])])
    market_plus_oof_better=0
    comparison_count=0
    for delays in result['economics'].values():
        for variants in delays.values():
            baseline=variants['market_only']['independent']['pnl_per_turnover']
            augmented=variants['market_plus_oof']['independent']['pnl_per_turnover']
            if baseline is not None and augmented is not None:
                comparison_count+=1
                market_plus_oof_better+=int(augmented>baseline)
    latency_zero_context={delay:result['economics']['0'][str(delay)]['market_plus_oof_context']['portfolio']['final_balance']
                         for delay in EXECUTION_DELAYS}
    context_latency_zero_positive=all(value>100 for value in latency_zero_context.values())
    context_other_latency_positive=any(value['portfolio']['final_balance']>100
        for latency,delays in result['economics'].items() if latency!='0'
        for value in [delays[str(delay)]['market_plus_oof_context'] for delay in EXECUTION_DELAYS])
    lines+=_mdtable(['OOF latency s','Extra exec s','Variant','A bets','A PnL','A expected','A PnL/turnover','B bets','B final USD','B DD','A rejection reasons'],rows)+['',
        f'Economic comparison: adding OOF improved fixed-stake PnL/turnover in {market_plus_oof_better} of {comparison_count} paired latency/delay comparisons, with improvements confined to source latency 0s and execution delays +1s/+2s. The context-augmented model finished above $100 at all three execution delays for source latency 0s ({"yes" if context_latency_zero_positive else "no"}) and at another source latency ({"yes" if context_other_latency_positive else "no"}); that pattern is latency-sensitive and its paired score intervals include zero.',
        'At execution delays, the model chooses side and limit from the initial quote only. A later ask above that limit, missing forward quote, invalid book, insufficient current top ask shares, minimum-order failure or portfolio cash shortage rejects the order. The observed top ask size is not carried forward. Market/sample recording frequency does not establish exchange-book age.',
        'The 100 USD portfolio carries locked capital and official settlement across all folds. Drawdown is based on cash plus locked cost, not mark-to-market. Independent fixed-stake results continue after hypothetical bankroll exhaustion and are not a feasible 100 USD portfolio. Aggregate and per-fold trade counts, fees, expected/realized PnL and hit rates are saved in JSON; full quote-level rejection details are in the Parquet ledgers.',
        'Obadiaha remains excluded from this price-conditioned experiment. Its raw source, earlier audit and findings remain preserved.', '',
        'Reproduction (pinned local caches; no source retraining):','','```powershell','python build_polymarket_history.py',
        'python -m unittest discover -s tests -p test_polymarket_market_value.py',
        'python -m unittest discover -s tests -p test_polymarket_history.py',
        'python -m unittest discover -s tests -p test_polymarket_adaptation.py','```','',
        'All variants are research only. Live settings and run.py are unchanged.']
    path.write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')


def _mdtable(header,rows):
    return (['| '+' | '.join(map(str,header))+' |','| '+' | '.join(['---']*len(header))+' |']+
            ['| '+' | '.join(map(str,r))+' |' for r in rows])
