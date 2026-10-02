"""Chronological calibration and economic policies, independent of data ingestion."""
import heapq
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from utils.polymarket import polymarket_taker_fee_fraction_of_notional
from utils.polymarket_history import fee_model, payoff, write_json
from utils.trading import (build_trade_intent, decide_trade_from_ev, decide_trade_from_model_direction,
                          load_trade_policy_runtime_config, _minimum_executable_stake_usdc)
from utils.project_config import load_runtime_artifact_paths, load_runtime_asset_settings, load_live_profile

INITIAL_BANKROLL = 100.0
MIN_ISOTONIC_ROWS = 1000
BASELINES = [
    {'name': 'no_trade', 'kind': 'none'},
    {'name': 'current_live_ev', 'kind': 'live'},
    {'name': 'fixed_5', 'kind': 'fixed', 'stake': 5},
    {'name': 'fixed_10_edge_02', 'kind': 'fixed', 'stake': 10, 'edge': .02},
    {'name': 'fixed_10_edge_05', 'kind': 'fixed', 'stake': 10, 'edge': .05},
    {'name': 'fraction_01', 'kind': 'fraction', 'fraction': .01},
    {'name': 'fraction_05', 'kind': 'fraction', 'fraction': .05},
    {'name': 'full_kelly', 'kind': 'kelly', 'fraction': 1},
    {'name': 'half_kelly', 'kind': 'kelly', 'fraction': .5},
    {'name': 'quarter_kelly', 'kind': 'kelly', 'fraction': .25},
    {'name': 'capped_kelly', 'kind': 'kelly', 'fraction': 1, 'cap_fraction': .05},
    {'name': 'kelly_edge_02', 'kind': 'kelly', 'fraction': .5, 'edge': .02},
]


def logits(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(past, method, evaluation_start):
    if not (past.resolved_at_utc < evaluation_start).all():
        raise ValueError('Calibration includes unsettled/future labels')
    if method == 'none':
        return None
    if len(past) < 100 or past.target_polymarket_up.nunique() < 2:
        raise ValueError('Insufficient calibration classes/observations')
    if method == 'platt':
        return LogisticRegression(C=1e6, max_iter=1000).fit(logits(past.p_model_up), past.target_polymarket_up)
    if method == 'isotonic' and len(past) >= MIN_ISOTONIC_ROWS:
        return IsotonicRegression(out_of_bounds='clip', y_min=1e-6, y_max=1-1e-6).fit(past.p_model_up, past.target_polymarket_up)
    raise ValueError('Unknown method or insufficient isotonic rows')


def calibrate(model, p):
    if model is None:
        return np.asarray(p, dtype=float)
    if isinstance(model, LogisticRegression):
        return model.predict_proba(logits(p))[:, 1]
    return model.predict(p)


def calibration_metrics(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
    frame = pd.DataFrame({'p': p, 'y': np.asarray(y, dtype=float)})
    frame['bin'] = pd.cut(frame.p, np.linspace(0, 1, 11), include_lowest=True)
    bins = [{'bin': str(bucket), 'count': len(g), 'probability': float(g.p.mean()),
             'actual_up_rate': float(g.y.mean())} for bucket, g in frame.groupby('bin', observed=True)]
    slope, intercept = None, None
    if len(np.unique(y)) == 2 and np.std(p) > 1e-8:
        diagnostic = LogisticRegression(C=1e6, max_iter=1000).fit(logits(p), y)
        slope, intercept = float(diagnostic.coef_[0, 0]), float(diagnostic.intercept_[0])
    return {'brier_score': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'slope': slope, 'intercept': intercept, 'reliability_bins': bins}


def chronological_splits(data, folds=3):
    ordered = data.sort_values('decision_available_at')
    times = ordered.decision_available_at.drop_duplicates().to_numpy()
    edges = np.linspace(.4, 1, folds + 1)
    for fold in range(folds):
        lo = times[min(int(len(times) * edges[fold]), len(times) - 1)]
        hi = times[min(int(len(times) * edges[fold + 1]), len(times) - 1)]
        past = ordered[(ordered.decision_available_at < lo) & (ordered.resolved_at_utc < lo)]
        future = ordered[(ordered.decision_available_at >= lo) &
                         ((ordered.decision_available_at <= hi) if fold == folds-1 else (ordered.decision_available_at < hi))]
        yield fold, past, future


def optimal_kelly_stake(p, price, fees, bankroll, ask_size, minimum_shares, return_reason=False):
    """Numerically maximize actual binary net payoff with existing rounded fee math."""
    def result(stake, reason):
        return (stake, reason) if return_reason else stake

    cap = min(bankroll * .999, ask_size * price)
    if not np.isfinite(cap) or cap <= 0:
        return result(0., 'insufficient_liquidity')

    def objective(stake):
        win = payoff(stake, price, 1, fees, ask_size)
        if win is None:
            return np.inf
        return -(p * np.log1p(win['pnl'] / bankroll) + (1-p) * np.log1p(-stake / bankroll))

    optimum = minimize_scalar(objective, bounds=(.00001, cap), method='bounded')
    candidate = float(optimum.x)
    won = payoff(candidate, price, 1, fees, ask_size)
    if not won:
        return result(0., 'insufficient_liquidity')
    if won['shares'] < minimum_shares:
        return result(0., 'minimum_order')
    if objective(candidate) >= 0:
        return result(0., 'no_positive_log_growth')
    return result(candidate, 'sized')


def policy_action(row, probability, cash, equity, config, live_config, exposure=0, return_reason=False):
    def reject(reason):
        return (None, reason) if return_reason else None

    if config['kind'] == 'none':
        return reject('policy_no_trade')
    fees = fee_model(row)
    prices = {'up': row.up_best_ask, 'down': row.down_best_ask}
    sizes = {'up': row.up_ask_size, 'down': row.down_ask_size}
    if any(not np.isfinite(x) or not 0 < x < 1 for x in prices.values()):
        return reject('invalid_price')
    fractions = {side: polymarket_taker_fee_fraction_of_notional(price, fees) for side, price in prices.items()}
    kwargs = dict(proba_up=probability, ask_yes=prices['up'], ask_no=prices['down'],
                  fee_yes=fractions['up'], fee_no=fractions['down'],
                  extra_buffer=live_config['extra_buffer'] if config['kind'] == 'live' else config.get('edge', 0))
    if config['kind'] != 'live' or config.get('exact_fee_entry', False):
        # Live deducts fees from gross stake; its economic break-even probability
        # is ask/(1-fee_fraction), rather than ask + fee_fraction.
        if any(value >= 1 for value in fractions.values()):
            return reject('invalid_fee')
        kwargs['fee_yes'] = prices['up'] / (1-fractions['up']) - prices['up']
        kwargs['fee_no'] = prices['down'] / (1-fractions['down']) - prices['down']
    if config['kind'] == 'live' and live_config['mode'] == 'model_direction_min_stake':
        decision = decide_trade_from_model_direction(**kwargs, threshold=live_config.get('threshold', .5),
            **{k: live_config[k] for k in ['min_decision_margin', 'min_decision_margin_up', 'min_decision_margin_down'] if k in live_config})
    else:
        decision = decide_trade_from_ev(**kwargs)
    if decision['decision'] == 'no_trade':
        return reject('no_positive_edge')
    side = 'up' if decision['decision'] == 'buy_yes' else 'down'
    price, available = prices[side], sizes[side]
    if config['kind'] == 'live' and (row.seconds_to_expiry <= live_config.get('no_trade_last_seconds', 20)
                                    or price > live_config.get('order_price_cap', .95)):
        return reject('live_price_or_expiry_limit')
    if config.get('quality_margin', 0):
        # A sampled cached book has unknown exchange age. Use only observed spread,
        # never later snapshots, as a conservative execution uncertainty margin.
        spread = max(row.up_best_ask-row.up_best_bid, row.down_best_ask-row.down_best_bid)
        margin = config['quality_margin'] * spread
        p_side = probability if side == 'up' else 1-probability
        if p_side - price / (1-fractions[side]) <= config.get('edge', 0) + margin:
            return reject('quote_quality_margin')
    if not np.isfinite(available) or available <= 0:
        return reject('insufficient_liquidity')
    if cash <= 0:
        return reject('locked_capital' if exposure > 0 else 'no_cash')
    if config['kind'] == 'live':
        multiplier = live_config['stake_multiplier']
        intent = build_trade_intent(policy_result=decision, bankroll=cash, stake_multiplier=1 if multiplier == 'return_multiple' else multiplier,
                                    fee_model=fees, order_min_size=row.order_min_size,
                                    external_stake_cap_usdc=max(0, live_config.get('max_exposure_usdc', 100) - exposure),
                                    stake_multiplier_mode=live_config.get('stake_multiplier_mode', 'return_multiple' if multiplier == 'return_multiple' else 'fixed'),
                                    initial_bankroll=INITIAL_BANKROLL, return_multiple_balance=cash)
        if intent['decision'] == 'no_trade':
            if intent['final_reason'] == 'stake_above_bankroll':
                required = intent['effective_stake_usdc']
                if exposure > 0 and required <= equity:
                    return reject('locked_capital')
                return reject('minimum_order_unaffordable' if intent['required_stake_usdc'] > cash else 'insufficient_cash')
            if intent['final_reason'] == 'shares_below_order_min':
                return reject('minimum_order')
            return reject('live_'+intent['final_reason'])
        stake = intent['bet_usdc']
    elif config['kind'] == 'fixed':
        stake = config['stake']
    elif config['kind'] == 'fraction':
        stake = equity * config['fraction']
    else:
        p = probability if side == 'up' else 1 - probability
        optimum, sizing_reason = optimal_kelly_stake(p, price, fees, equity, available, row.order_min_size, True)
        if optimum <= 0:
            return reject(sizing_reason)
        stake = optimum * config['fraction']
        stake = min(stake, equity * config.get('cap_fraction', 1))
    stake = round(stake, 2)
    if stake <= 0:
        minimum = _minimum_executable_stake_usdc(entry_price=price, fee_model=fees, order_min_size=row.order_min_size)
        if minimum > available*price:
            return reject('insufficient_liquidity')
        return reject('sizing_zero')
    if stake > cash:
        return reject('locked_capital' if exposure > 0 and stake <= equity else 'insufficient_cash')
    result = payoff(stake, price, 1, fees, available)
    if result is None:
        return reject('insufficient_liquidity')
    if result['shares'] < row.order_min_size:
        return reject('minimum_order')
    p_side = probability if side == 'up' else 1-probability
    expected = p_side * result['shares'] - stake
    if config.get('exact_fee_entry', False) and expected <= 0:
        return reject('nonpositive_exact_expected_pnl')
    action = dict(side=side, stake=stake, price=price, shares=result['shares'], fee=result['fee'],
                  edge=decision['ev_yes'] if side == 'up' else decision['ev_no'],
                  p_side=p_side, expected_pnl=expected,
                  break_even_probability=stake/result['shares'],
                  probability_edge=p_side-stake/result['shares'])
    return (action, 'executed') if return_reason else action


def backtest(data, config, live_config, initial_bankroll=INITIAL_BANKROLL, independent_funding=False):
    if independent_funding and config['kind'] != 'fixed':
        raise ValueError('Independent funding is fixed-stake signal diagnostics only')
    cash, locked, equity = initial_bankroll, 0., initial_bankroll
    pending, path, trades, seen = [], [], [], set()
    sequence, realized_pnl, unable_minimum = 0, 0., 0
    rejection_reasons = Counter()

    def snapshot(timestamp, event, cash_flow=0., cannot_order=False):
        return {'timestamp_utc': timestamp, 'bankroll': cash + locked,
                'cost_basis_equity': cash + locked, 'cash': cash, 'exposure': locked,
                'open_position_cost': locked, 'open_position_market_value': np.nan if pending else 0.,
                'mark_to_market_equity': np.nan if pending else cash,
                'realized_pnl': realized_pnl, 'cash_flow': cash_flow,
                'cannot_place_minimum_order': cannot_order, 'event': event}

    def settle(until):
        nonlocal cash, locked, equity, realized_pnl
        while pending and pending[0][0] <= until:
            timestamp, _, stake, payout = heapq.heappop(pending)
            cash += payout
            locked -= stake
            locked = max(0., locked) if pending else 0.
            realized_pnl += payout - stake
            equity = cash + locked
            path.append(snapshot(timestamp, 'settlement', payout))

    ordered = data.assign(execution_at=data[['decision_available_at', 'timestamp_utc']].max(axis=1))
    for row in ordered.sort_values('execution_at').itertuples():
        settle(row.execution_at)
        if row.condition_id in seen:
            continue
        active = next(c for c in BASELINES if c['name'] == row.selected_policy) if config['kind'] == 'selected' else config
        if getattr(row, 'research_quality_selected', False) and config['kind'] == 'selected':
            active = dict(active, quality_margin=row.selected_quality_margin, exact_fee_entry=True,
                          adapted_probability=True)
        probability = (getattr(row, 'p_model_up', row.p_calibrated)
                       if active['kind'] == 'live' and not active.get('adapted_probability', False)
                       else row.p_calibrated)
        minimums = [_minimum_executable_stake_usdc(entry_price=getattr(row, side + '_best_ask'),
                    fee_model=fee_model(row), order_min_size=row.order_min_size) for side in ['up', 'down']]
        cannot_order = not any(np.isfinite(stake) and stake > 0 and stake <= cash for stake in minimums)
        unable_minimum += int(cannot_order)
        action, reason = policy_action(row, probability, initial_bankroll if independent_funding else cash,
                                       initial_bankroll if independent_funding else equity,
                                       active, live_config, 0 if independent_funding else locked, return_reason=True)
        rejection_reasons[reason] += 1
        if action:
            stake = action['stake']
            outcome = row.target_polymarket_up if action['side'] == 'up' else 1-row.target_polymarket_up
            action['pnl'] = action['shares'] * outcome - stake
            action['return'] = action['pnl'] / stake
            cash -= stake
            locked += stake
            sequence += 1
            heapq.heappush(pending, (max(row.resolved_at_utc, row.market_end_utc), sequence, stake, stake + action['pnl']))
            seen.add(row.condition_id)
            trades.append(dict(action, condition_id=row.condition_id, timestamp_utc=row.timestamp_utc,
                               decision_available_at=row.decision_available_at, probability=probability,
                               policy_name=active.get('name', active['kind']),
                               fold_id=getattr(row, 'evaluation_fold', None),
                               source=getattr(row, 'source', None),
                               settlement_available_at=max(row.resolved_at_utc, row.market_end_utc),
                               p_model_up=getattr(row, 'p_model_up', probability), p_calibrated=row.p_calibrated,
                               seconds_to_expiry=row.seconds_to_expiry, quote_delay_ms=row.quote_delay_ms))
        path.append(dict(snapshot(row.execution_at, 'decision', -action['stake'] if action else 0., cannot_order),
                         rejection_reason=reason))
    settle(pd.Timestamp.max.tz_localize('UTC'))
    tr = pd.DataFrame(trades)
    curve = pd.DataFrame(path)
    balances = np.r_[initial_bankroll, curve.bankroll.to_numpy() if len(curve) else []]
    drawdown = balances / np.maximum.accumulate(balances) - 1
    if (cash < -1e-8 and not independent_funding) or not np.isclose(cash - initial_bankroll, realized_pnl):
        raise AssertionError('Portfolio cash/PnL conservation broken')
    summary = {'initial_bankroll': initial_bankroll, 'final_balance': cash,
               'net_pnl': equity-initial_bankroll, 'return_multiple': equity/initial_bankroll,
               'realized_pnl': realized_pnl, 'final_open_position_cost': locked,
               'drawdown_valuation': 'cash plus open-position cost; not mark-to-market',
               'minimum_order_unaffordable_decisions': unable_minimum,
               'log_growth': float(np.log(equity/initial_bankroll)) if equity > 0 else None,
               'max_drawdown': float(-drawdown.min()), 'number_trades': len(tr),
               'fraction_markets_traded': len(seen)/max(data.condition_id.nunique(), 1),
               'turnover': float(tr.stake.sum()) if len(tr) else 0.,
               'average_stake': float(tr.stake.mean()) if len(tr) else None,
               'average_entry_price': float(tr.price.mean()) if len(tr) else None,
               'up_trades': int(tr.side.eq('up').sum()) if len(tr) else 0,
               'down_trades': int(tr.side.eq('down').sum()) if len(tr) else 0,
               'hit_rate': float(tr.pnl.gt(0).mean()) if len(tr) else None,
               'average_edge_at_entry': float(tr.edge.mean()) if len(tr) else None,
               'pnl_per_trade': float(tr.pnl.mean()) if len(tr) else None,
               'worst_trade': float(tr.pnl.min()) if len(tr) else None,
               'max_exposure': float(curve.exposure.max()) if len(curve) else 0.,
               'execution_rejections_or_no_trade': len(data)-len(tr)}
    summary.update(rejection_reasons=dict(rejection_reasons),
                   funding='independent bets; not a feasible $100 portfolio' if independent_funding else 'continuous $100 portfolio',
                   fees=float(tr.fee.sum()) if len(tr) else 0.,
                   expected_pnl=float(tr.expected_pnl.sum()) if len(tr) else 0.,
                   pnl_per_turnover=float(tr.pnl.sum()/tr.stake.sum()) if len(tr) else None,
                   negative_expected_pnl_trades=int(tr.expected_pnl.le(0).sum()) if len(tr) else 0)
    if independent_funding:
        summary.update(initial_bankroll=None, final_balance=None, return_multiple=None,
                       log_growth=None, max_drawdown=None,
                       minimum_order_unaffordable_decisions=None)
    buckets = {}
    if len(tr):
        boundaries = {'price': [0, .2, .4, .6, .8, 1], 'p_side': [0, .4, .5, .6, 1],
                      'edge': [-np.inf, 0, .01, .02, .05, .1, np.inf],
                      'seconds_to_expiry': [0, 60, 120, 240, 300],
                      'quote_delay_ms': [-.01, 0, 500, 1000, 2000, np.inf]}
        for col, breaks in boundaries.items():
            buckets[col] = [{'bucket': str(b), 'trades': len(g), 'pnl': float(g.pnl.sum()), 'turnover': float(g.stake.sum())}
                            for b, g in tr.groupby(pd.cut(tr[col], breaks, include_lowest=True), observed=True)]
    summary['buckets'] = buckets
    return summary, tr, curve


def evaluate_walk_forward(data, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime_asset_settings('BTC')
    artifacts = load_runtime_artifact_paths(asset='BTC')
    live_config = load_trade_policy_runtime_config(artifacts['trade_policy_path'], asset='BTC')
    live_profile = load_live_profile(runtime['live_profile'], dataset_asset='BTC')
    live_config['threshold'] = json.loads(Path(artifacts['model_meta_path']).read_text())['prediction_threshold']
    live_config.update({key: live_profile['polymarket_' + key] for key in
                        ['no_trade_last_seconds', 'order_price_cap', 'max_exposure_usdc']})
    reports, future_blocks = [], []
    for fold, past, future in chronological_splits(data):
        # Inner calibration training -> policy selection. Outer future stays untouched.
        cutoff = past.decision_available_at.iloc[int(len(past)*.6)]
        calibration_past = past[past.resolved_at_utc < cutoff]
        selection = past[past.decision_available_at >= cutoff].copy()
        choices = ['none']
        if len(calibration_past) >= 100 and calibration_past.target_polymarket_up.nunique() == 2:
            choices += ['platt'] + (['isotonic'] if len(calibration_past) >= MIN_ISOTONIC_ROWS else [])
        calibration_selection = {}
        for method in choices:
            model = fit_calibrator(calibration_past, method, selection.decision_available_at.min())
            p = calibrate(model, selection.p_model_up)
            calibration_selection[method] = calibration_metrics(selection.target_polymarket_up, p)
        selected_method = min(choices, key=lambda m: calibration_selection[m]['log_loss'])
        inner_model = fit_calibrator(calibration_past, selected_method, selection.decision_available_at.min())
        selection['p_calibrated'] = calibrate(inner_model, selection.p_model_up)
        selection_results = {c['name']: backtest(selection, c, live_config)[0] for c in BASELINES}
        selected_policy = max(BASELINES, key=lambda c: selection_results[c['name']]['log_growth']
                              if selection_results[c['name']]['log_growth'] is not None else -np.inf)
        calibration_future, model = {}, None
        for method in choices:
            fitted = fit_calibrator(past, method, future.decision_available_at.min())
            calibrated = calibrate(fitted, future.p_model_up)
            calibration_future[method] = calibration_metrics(future.target_polymarket_up, calibrated)
            if method == selected_method:
                model = fitted
        future = future.copy()
        future['p_calibrated'] = calibrate(model, future.p_model_up)
        future['evaluation_fold'] = fold
        future['selected_policy'] = selected_policy['name']
        future_blocks.append(future)
        future.to_parquet(destination / f'fold_{fold}_calibrated.parquet', index=False)
        report = {'fold': fold, 'calibration_train_rows': len(calibration_past), 'selection_rows': len(selection),
                  'past_rows': len(past), 'evaluation_rows': len(future),
                  'evaluation_start_utc': str(future.decision_available_at.min()),
                  'evaluation_end_utc': str(future.decision_available_at.max()),
                  'calibration_labels_available_before': str(past.resolved_at_utc.max()),
                  'selected_calibration': selected_method, 'selected_policy': selected_policy['name'],
                  'selection_calibration': calibration_selection, 'future_calibration': calibration_future,
                  'policy_selection': selection_results,
                  'live_replication_limits': 'Same EV/direction and build_trade_intent sizing; historical feeSchedule, ask and observed size. Current tick slippage is an order limit, not an assumed fill. No live withdrawal cap, redeem delays or actual fills reconstructed.',
                  'bankroll_contract': 'One continuous $100 portfolio per variant/scenario across all outer folds; settlement available at max(official resolution, expiry), no redemption delay. Cost basis equity is not mark-to-market.'}
        write_json(destination / f'fold_{fold}_report.json', report)
        reports.append(report)
        print(f'Policy fold {fold}: {len(future)} future markets, selected {selected_policy["name"]}', flush=True)
    continuous = pd.concat(future_blocks, ignore_index=True)
    result = {'folds': reports, 'continuous_baselines': {},
              'evaluation_rows': len(continuous),
              'evaluation_start_utc': str(continuous.decision_available_at.min()),
              'evaluation_end_utc': str(continuous.decision_available_at.max()),
              'evaluation_source_coverage': {source: {'rows': len(g),
                  'start_utc': str(g.decision_available_at.min()), 'end_utc': str(g.decision_available_at.max())}
                  for source, g in continuous.groupby('source')},
              'live_config': live_config}
    for config in BASELINES + [{'name': 'past_selected_policy', 'kind': 'selected'}]:
        name = config['name']
        summary, trades, path = backtest(continuous, config, live_config)
        trades.to_parquet(destination / (name + '_trades.parquet'), index=False)
        path.to_parquet(destination / (name + '_bankroll.parquet'), index=False)
        result['continuous_baselines'][name] = summary
    write_json(destination / 'walk_forward.json', result)
    return result
