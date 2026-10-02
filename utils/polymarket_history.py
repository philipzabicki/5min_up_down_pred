"""Revision-pinned downloads and auditable historical Polymarket contracts."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.polymarket import (
    parse_json_listish, polymarket_fee_model_from_market,
    resolve_polymarket_up_down_tokens as token_mapping,
    polymarket_market_slug_matches_prefix,
    resolve_polymarket_actual_up_from_market_payload,
    polymarket_taker_fee_usdc_from_notional,
)

PRIMARY = 'kachoio/polymarket-5-minute-crypto-up-down-markets'
SECONDARY = 'obadiaha/polymarket-crypto-5m-15m'
PRICE_COLS = ['up_best_bid', 'up_best_ask', 'down_best_bid', 'down_best_ask']


def utc(values):
    parsed = pd.to_datetime(values, utc=True, errors='raise')
    if isinstance(parsed, pd.Series):
        return parsed.dt.as_unit('ns')
    return parsed.as_unit('ns') if hasattr(parsed, 'as_unit') else parsed


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def checksum(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def session():
    client = requests.Session()
    client.mount('https://', HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    return client


def download_source(repo, files, root, revision=None):
    client = session()
    if revision is None:
        response = client.get('https://huggingface.co/api/datasets/' + repo, timeout=60)
        response.raise_for_status()
        revision = response.json()['sha']
    folder = Path(root) / repo.split('/')[0] / revision
    old_path = folder / 'manifest.json'
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    previous = {item['file']: item for item in old.get('files', [])}
    records = []
    for name in files:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        prior = previous.get(name)
        if path.exists() and prior and checksum(path) != prior['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')
        if not path.exists():
            part = path.with_suffix(path.suffix + '.part')
            offset = part.stat().st_size if part.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with client.get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}',
                            headers=headers, stream=True, timeout=120) as r:
                r.raise_for_status()
                append = offset > 0 and r.status_code == 206
                if append and not r.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Unexpected range response')
                with part.open('ab' if append else 'wb') as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
            part.replace(path)
        records.append({'file': name, 'size_bytes': path.stat().st_size,
                        'sha256': checksum(path),
                        'downloaded_at_utc': prior['downloaded_at_utc'] if prior else pd.Timestamp(path.stat().st_mtime, unit='s', tz='UTC').isoformat()})
    manifest = {'source': repo, 'revision': revision, 'instrument': 'BTC',
                'downloaded_at_utc': old.get('downloaded_at_utc', pd.Timestamp.now(tz='UTC').isoformat()),
                'files': records, 'table_rows': {}}
    instruments = set()
    for name in files:
        if not name.endswith('.parquet'):
            continue
        f = pq.ParquetFile(folder / name)
        if 'asset' in f.schema.names:
            for batch in f.iter_batches(columns=['asset']):
                instruments.update(batch.column(0).to_pylist())
        manifest.setdefault('table_rows', {})[name] = f.metadata.num_rows
        timestamp = next((c for c in ['ts_utc', 'timestamp', 'market_start', 'start_time'] if c in f.schema.names), None)
        if timestamp:
            lo, hi = None, None
            for batch in f.iter_batches(columns=[timestamp]):
                t = utc(batch.to_pandas()[timestamp]).dropna()
                if len(t):
                    lo = t.min() if lo is None else min(lo, t.min())
                    hi = t.max() if hi is None else max(hi, t.max())
            manifest.setdefault('coverage', {})[name] = {'start_utc': str(lo), 'end_utc': str(hi)}
    manifest['number_markets'] = manifest['table_rows'].get('btc_markets.parquet', manifest['table_rows'].get('markets/all.parquet', 0))
    manifest['number_observations'] = sum(v for k, v in manifest['table_rows'].items() if 'ticks' in k or 'orderbooks' in k)
    manifest['instrument'] = sorted(instruments) if instruments else 'BTC'
    write_json(old_path, manifest)
    return folder


def cache_official(markets, root, workers=12):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        path = root / (row.market_slug + '.json')
        if path.exists():
            return
        client = session()
        response = client.get('https://gamma-api.polymarket.com/markets/slug/' + row.market_slug, timeout=45)
        if response.status_code == 404:
            write_json(path, {'fetched_at_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'payload': None})
        else:
            response.raise_for_status()
            write_json(path, {'fetched_at_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'payload': response.json()})
        client.close()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(fetch, markets.itertuples()), 1):
            if i % 1000 == 0:
                print(f'Official cache: {i}/{len(markets)}', flush=True)
    records = []
    for slug in markets.market_slug:
        path = root / (slug + '.json')
        records.append({'file': path.name, 'size_bytes': path.stat().st_size, 'sha256': checksum(path)})
    # Scope-specific manifests do not overwrite the independently cached secondary collection.
    scope = hashlib.sha256('\n'.join(sorted(markets.market_slug)).encode()).hexdigest()[:12]
    epochs = [int(slug.rsplit('-', 1)[1]) for slug in markets.market_slug]
    write_json(root / f'manifest_{scope}.json', {
        'source': 'official_gamma', 'revision': 'API snapshot; per-file fetched_at_utc',
        'downloaded_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'files': records, 'number_markets': len(markets), 'number_observations': 0,
        'instrument': 'BTC', 'coverage_start_utc': str(pd.Timestamp(min(epochs), unit='s', tz='UTC')),
        'coverage_end_utc': str(pd.Timestamp(max(epochs)+300, unit='s', tz='UTC'))})


def normalize_markets(raw, official_root):
    m = raw.rename(columns={'slug': 'market_slug', 'token_up': 'up_token_id',
                            'token_down': 'down_token_id', 'market_start': 'market_start_utc',
                            'market_end': 'market_end_utc', 'outcome': 'vendor_inferred_outcome'}).copy()
    for c in ['market_start_utc', 'market_end_utc']:
        m[c] = utc(m[c])
    m['asset'], m['source'] = 'BTC', PRIMARY
    m['polymarket_outcome_up'] = np.nan
    m['resolved_at_utc'] = pd.NaT
    m['validation_status'] = 'missing_official'
    m['token_mapping_valid'] = False
    m['outcome_source'] = 'unavailable'
    m['fee_rate'], m['fee_exponent'] = np.nan, np.nan
    m['fee_round_decimals'], m['fee_min_fee'] = 5, 0.00001
    m['order_min_size'], m['tick_size'] = np.nan, np.nan
    for row in m.itertuples():
        path = Path(official_root) / (row.market_slug + '.json')
        if not path.exists():
            continue
        payload = json.loads(path.read_text())['payload']
        if not payload:
            continue
        try:
            mapping = token_mapping(payload)
            valid = (str(payload.get('conditionId')) == row.condition_id and
                     payload.get('slug') == row.market_slug and
                     mapping['up'] == row.up_token_id and mapping['down'] == row.down_token_id)
            start = payload.get('eventStartTime') or next((e.get('startTime') for e in payload.get('events', []) if e.get('startTime')), None)
            epoch = pd.Timestamp(int(row.market_slug.rsplit('-', 1)[1]), unit='s', tz='UTC')
            valid = valid and start is not None and utc(start) == row.market_start_utc == epoch
            valid = valid and int(epoch.timestamp()) % 300 == 0 and polymarket_market_slug_matches_prefix(row.market_slug, "btc-updown-5m")
            valid = valid and utc(payload['endDate']) == row.market_end_utc == epoch + pd.Timedelta(minutes=5)
            if not valid:
                m.loc[row.Index, 'validation_status'] = 'identity_or_boundary_mismatch'
                continue
            m.loc[row.Index, 'token_mapping_valid'] = True
            # Protect the shared live resolver against near-terminal open-book prices.
            resolved = payload.get('closed') and str(payload.get('umaResolutionStatus', '')).lower() == 'resolved'
            final_prices = [float(x) for x in parse_json_listish(payload.get('outcomePrices'))]
            unambiguous = sorted(final_prices) == [0., 1.]
            outcome = resolve_polymarket_actual_up_from_market_payload(payload) if resolved and unambiguous else None
            fees = polymarket_fee_model_from_market(payload)
            m.loc[row.Index, 'fee_source'] = fees['source']
            for key in ['rate', 'exponent', 'round_decimals', 'min_fee']:
                m.loc[row.Index, 'fee_' + key] = fees['fee_round_decimals' if key == 'round_decimals' else key]
            m.loc[row.Index, ['order_min_size', 'tick_size']] = [payload.get('orderMinSize'), payload.get('orderPriceMinTickSize')]
            m.loc[row.Index, 'polymarket_outcome_up'] = np.nan if outcome is None else outcome
            times = [utc(payload[k]) for k in ['umaEndDate', 'closedTime'] if payload.get(k)]
            resolved_at = max(times) if times else None
            if resolved_at is not None and resolved_at < row.market_end_utc:
                m.loc[row.Index, 'polymarket_outcome_up'] = np.nan
                m.loc[row.Index, 'validation_status'] = 'invalid_settlement_time'
                continue
            m.loc[row.Index, 'resolved_at_utc'] = resolved_at.tz_localize(None) if resolved_at else pd.NaT
            m.loc[row.Index, 'validation_status'] = ('validated' if outcome is not None and resolved_at else
                                                    'missing_resolution_time' if outcome is not None else 'unresolved')
            m.loc[row.Index, 'outcome_source'] = 'official_gamma' if outcome is not None else 'unavailable'
        except (ValueError, KeyError, TypeError):
            m.loc[row.Index, 'validation_status'] = 'invalid_metadata'
    m['resolved_at_utc'] = utc(m['resolved_at_utc'])
    m['duplicate_market'] = m.condition_id.duplicated(keep=False) | m.market_start_utc.duplicated(keep=False)
    return m


def normalize_quotes(raw, markets, previous=None):
    q = raw.rename(columns={'ts_utc': 'timestamp_utc', 'bu': 'up_best_bid', 'au': 'up_best_ask',
                            'bd': 'down_best_bid', 'ad': 'down_best_ask', 'su': 'up_bid_size',
                            'sd': 'down_bid_size', 'sau': 'up_ask_size', 'sad': 'down_ask_size',
                            'du': 'up_bid_depth_5c_usdc', 'dd': 'down_bid_depth_5c_usdc'}).copy()
    q['timestamp_utc'] = utc(q['timestamp_utc'])
    q['timestamp_disagreement'] = q.timestamp_utc.ne(pd.to_datetime(q.t, unit='s', utc=True)) if 't' in q else False
    q = q.merge(markets[['condition_id', 'market_start_utc', 'market_end_utc']], on='condition_id', how='left', validate='many_to_one')
    q['seconds_to_expiry'] = (q.market_end_utc - q.timestamp_utc).dt.total_seconds()
    q['outside_market'] = (q.timestamp_utc < q.market_start_utc) | (q.timestamp_utc >= q.market_end_utc) | q.market_end_utc.isna()
    q['missing_side'] = q[PRICE_COLS].isna().any(axis=1)
    q['invalid_price'] = ((q[PRICE_COLS] < 0) | (q[PRICE_COLS] > 1)).any(axis=1)
    q['crossed_book'] = (q.up_best_bid > q.up_best_ask) | (q.down_best_bid > q.down_best_ask)
    q['invalid_size'] = (q[['up_ask_size', 'down_ask_size']] < 0).any(axis=1)
    q['duplicate_timestamp'] = q.duplicated(['condition_id', 'timestamp_utc'], keep=False)
    delta = q.groupby('condition_id', sort=False).timestamp_utc.diff().dt.total_seconds()
    if previous is not None:
        first = ~q.condition_id.duplicated()
        before = q.loc[first, 'condition_id'].map(previous)
        delta.loc[first] = (q.loc[first, 'timestamp_utc'] - utc(before)).dt.total_seconds()
        q.loc[first & delta.eq(0), 'duplicate_timestamp'] = True
    q['non_monotonic_timestamp'] = delta.lt(0)
    q['observation_gap_seconds'] = delta
    q['stale_observation'] = delta.gt(2)
    q['book_age_known'] = False
    q['asset'], q['source'] = 'BTC', PRIMARY
    q['quote_valid'] = ~q[['outside_market', 'missing_side', 'invalid_price', 'crossed_book',
                           'duplicate_timestamp', 'non_monotonic_timestamp', 'timestamp_disagreement', 'invalid_size']].any(axis=1)
    return q.drop(columns=['t', 'market_start_utc', 'market_end_utc'], errors='ignore')


def normalize_tape(path, markets, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer, previous = None, {}
    counts, intervals, suspect_markets = {}, [], set()
    start = time.perf_counter()
    try:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=100000):
            q = normalize_quotes(batch.to_pandas(), markets, previous)
            previous.update(q.groupby('condition_id').timestamp_utc.last().to_dict())
            suspect_markets.update(q.loc[q.duplicate_timestamp | q.non_monotonic_timestamp, 'condition_id'])
            table = pa.Table.from_pandas(q, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination.with_suffix('.tmp'), table.schema, compression='zstd')
            writer.write_table(table)
            for c in ['quote_valid', 'missing_side', 'crossed_book', 'duplicate_timestamp',
                      'invalid_price', 'invalid_size', 'outside_market', 'timestamp_disagreement',
                      'non_monotonic_timestamp', 'stale_observation']:
                counts[c] = counts.get(c, 0) + int(q[c].sum())
            for side in ['up', 'down']:
                for kind in ['ask', 'bid']:
                    key = f'missing_{side}_{kind}_size'
                    counts[key] = counts.get(key, 0) + int(q[f'{side}_{kind}_size'].isna().sum())
            counts['quote_count'] = counts.get('quote_count', 0) + len(q)
            intervals.append(q.observation_gap_seconds.dropna().to_numpy())
    finally:
        if writer:
            writer.close()
    destination.with_suffix('.tmp').replace(destination)
    values = np.concatenate(intervals)
    counts.update(median_quote_interval_s=float(np.median(values)), p95_quote_interval_s=float(np.quantile(values, .95)),
                  normalization_seconds=time.perf_counter() - start)
    counts['suspect_tick_markets'] = sorted(suspect_markets)
    return counts


def select_quotes(decisions, tape_path, max_delay_ms):
    """Bounded memory forward join: inspect candidates in batches, never nearest."""
    chosen = []
    decisions = decisions.copy()
    decisions['decision_available_at'] = utc(decisions.decision_available_at)
    ordered = decisions.sort_values('decision_available_at')
    wanted = set(decisions.condition_id)
    for batch in pq.ParquetFile(tape_path).iter_batches(batch_size=100000):
        q = batch.to_pandas()
        q = q[q.condition_id.isin(wanted)]
        if q.empty:
            continue
        q['timestamp_utc'] = utc(q.timestamp_utc)
        local = ordered[ordered.condition_id.isin(q.condition_id.unique())]
        joined = pd.merge_asof(local, q.sort_values('timestamp_utc'),
                               left_on='decision_available_at', right_on='timestamp_utc',
                               by='condition_id', direction='forward',
                               tolerance=pd.Timedelta(milliseconds=max_delay_ms), suffixes=('', '_quote'))
        chosen.append(joined[joined.timestamp_utc.notna()])
    if not chosen:
        result = decisions.assign(timestamp_utc=pd.Series(pd.NaT, index=decisions.index, dtype='datetime64[ns, UTC]'),
                                  quote_valid=False, quote_delay_ms=np.nan)
        for c in PRICE_COLS + ['up_ask_size', 'down_ask_size', 'seconds_to_expiry']:
            result[c] = np.nan
        return result
    matches = pd.concat(chosen).sort_values('timestamp_utc').drop_duplicates('decision_id', keep='first')
    extras = [c for c in matches if c not in decisions or c == 'decision_id']
    result = decisions.merge(matches[extras], on='decision_id', how='left', validate='one_to_one')
    result['quote_delay_ms'] = (result.timestamp_utc - result.decision_available_at).dt.total_seconds() * 1000
    if (result.quote_delay_ms.dropna() < 0).any():
        raise AssertionError('Lookahead contract broken')
    return result


def validate_oos(predictions):
    required = ['Opened', 'p_model_up']
    if any(c not in predictions for c in required):
        raise ValueError('Missing OOF timestamp or probability')
    times = utc(predictions.Opened)
    if times.isna().any() or times.duplicated().any():
        raise ValueError('Missing or duplicate UTC OOF timestamps')
    if not (times.dt.second.eq(0) & times.dt.microsecond.eq(0) & times.dt.nanosecond.eq(0)).all():
        raise ValueError('Opened must be a one-minute candle opening timestamp')
    if not (np.isfinite(predictions.p_model_up) & predictions.p_model_up.between(0, 1)).all():
        raise ValueError('Invalid OOF probabilities')
    # Main-model OOF construction, including early stopping, is an accepted input.
    # Validate optional evidence when supplied; do not demand nonexistent history.
    if 'is_oos' in predictions and not predictions.is_oos.eq(True).all():
        raise ValueError('Final-model predictions are not accepted')
    if 'fit_labels_available_at' in predictions and not (utc(predictions.fit_labels_available_at) < times).all():
        raise ValueError('Prediction fitted on future/unavailable labels')


def join_oos(predictions, markets, latency_seconds):
    validate_oos(predictions)
    p = predictions.copy()
    p['Opened'] = utc(p.Opened)
    p = p[p.Opened.dt.minute.mod(5).eq(4)]
    p['market_start_utc'] = p.Opened + pd.Timedelta(minutes=1)
    p['decision_available_at'] = p.market_start_utc + pd.Timedelta(seconds=latency_seconds)
    p['decision_id'] = p.Opened.astype(str) + f':latency={latency_seconds}'
    return p.merge(markets, on='market_start_utc', how='inner', validate='one_to_one')


def fee_model(row):
    return {'rate': float(row.fee_rate), 'exponent': float(row.fee_exponent),
            'fee_round_decimals': int(row.fee_round_decimals), 'min_fee': float(row.fee_min_fee),
            'source': 'historical_official_gamma'}


def payoff(stake, price, outcome, fees, ask_size=np.inf):
    if outcome not in (0, 1) or not np.isfinite(price) or not 0 < price < 1:
        return None
    fee = polymarket_taker_fee_usdc_from_notional(stake, price, fees)['fee_usdc']
    shares = (stake - fee) / price
    if np.isnan(ask_size) or ask_size < shares:
        return None
    pnl = shares * outcome - stake
    return {'pnl': pnl, 'return': pnl / stake, 'shares': shares, 'fee': fee, 'price': price}


def economic_dataset(joined, stakes=(5, 10, 25, 50, 100), utility_bankroll=1000):
    d = joined.copy()
    d['target_polymarket_up'] = d.polymarket_outcome_up
    d['target_mismatch'] = (d.target_binance_proxy_up != d.target_polymarket_up).where(d.target_polymarket_up.notna())
    d['eligible'] = (d.validation_status.eq('validated') & ~d.duplicate_market &
                     d.quote_valid.fillna(False) & d.target_polymarket_up.notna() &
                     d.resolved_at_utc.notna() & d.timestamp_utc.lt(d.market_end_utc))
    d['exclusion_reason'] = np.select(
        [mask.to_numpy(dtype=bool) for mask in [~d.validation_status.eq('validated'), d.duplicate_market,
         d.timestamp_utc.isna(), ~d.quote_valid.fillna(False).astype(bool), ~d.timestamp_utc.lt(d.market_end_utc)]],
        [d.validation_status.to_numpy(dtype=object), 'duplicate_market', 'missing_forward_quote', 'invalid_quote', 'quote_after_expiry'],
        default='eligible')
    d['NO_TRADE_pnl'], d['NO_TRADE_return'], d['NO_TRADE_log_growth'] = 0., 0., 0.
    d['utility_bankroll_usdc'] = float(utility_bankroll)
    for side in ['up', 'down']:
        for stake in stakes:
            prefix = f'BUY_{side.upper()}_{stake}'
            vals = []
            for row in d.itertuples():
                outcome = row.target_polymarket_up if side == 'up' else 1 - row.target_polymarket_up
                result = payoff(stake, getattr(row, side + '_best_ask'), outcome,
                                fee_model(row), getattr(row, side + '_ask_size')) if row.eligible else None
                vals.append(result)
            for field in ['pnl', 'return', 'price', 'fee']:
                d[prefix + '_' + field] = [v[field] if v else np.nan for v in vals]
            d[prefix + '_log_growth'] = np.log1p(d[prefix + '_pnl'] / utility_bankroll)
            d[prefix + '_available'] = [v is not None for v in vals]
    d['ask_vwap_beyond_top_available'] = False
    return d


def secondary_adapter(raw, markets):
    """Long token snapshots stay long: no asynchronous synthetic paired book."""
    q = raw[raw.asset.eq('BTC') & raw.market_id.str.startswith('btc-updown-5m-')].copy()
    q['timestamp_utc'] = utc(q.timestamp)
    ids = markets[['market_slug', 'condition_id', 'up_token_id', 'down_token_id']]
    q = q.merge(ids, left_on='market_id', right_on='market_slug', how='left', suffixes=('_vendor', ''))
    q['token_side'] = np.select([q.token_id.eq(q.up_token_id), q.token_id.eq(q.down_token_id)], ['up', 'down'], default='unknown')
    q['source'] = SECONDARY
    # Raw partitions preserve at most ten levels in expensive-to-cheap order.
    # Sorting those levels cannot establish the top of the original full book.
    q['best_prices_scope'] = 'stored_levels_only_full_book_unknown'
    # Preserve numeric L2 columns, never raw JSON in policy tables.
    for book in ['bid', 'ask']:
        parsed = q[book + '_levels'].map(lambda text: sorted(json.loads(text),
                                     key=lambda x: float(x['price']), reverse=book == 'bid'))
        for level in range(10):
            for field in ['price', 'size']:
                q[f'{book}_{field}_{level}'] = parsed.map(lambda xs: float(xs[level][field]) if len(xs) > level else np.nan)
    q['duplicate_timestamp'] = q.duplicated(['condition_id', 'token_id', 'timestamp_utc'], keep=False)
    return q.drop(columns=['bid_levels', 'ask_levels'])


def normalize_secondary_tapes(paths, markets, destination):
    """Stream every downloaded partition; retain separate token capture times."""
    writer, inventory = None, []
    previous, suspect = {}, set()
    destination = Path(destination)
    try:
        for path in paths:
            rows, lo, hi, slugs = 0, None, None, set()
            for batch in pq.ParquetFile(path).iter_batches(batch_size=100000):
                q = secondary_adapter(batch.to_pandas(), markets)
                if q.empty:
                    continue
                keys = pd.Series(list(zip(q.condition_id, q.token_id)), index=q.index)
                first = ~keys.duplicated()
                earlier = keys[first].map(previous)
                delta = q.groupby(['condition_id', 'token_id'], sort=False).timestamp_utc.diff()
                delta.loc[first] = q.loc[first, 'timestamp_utc'] - utc(earlier)
                q.loc[first & delta.eq(pd.Timedelta(0)), 'duplicate_timestamp'] = True
                q['non_monotonic_timestamp'] = delta.lt(pd.Timedelta(0))
                suspect.update(q.loc[q.duplicate_timestamp | q.non_monotonic_timestamp, 'condition_id'].dropna())
                previous.update(q.groupby(['condition_id', 'token_id']).timestamp_utc.last().to_dict())
                q['quote_valid'] = (q.token_side.ne('unknown') &
                    q.best_bid.between(0, 1) & q.best_ask.between(0, 1) & q.best_bid.le(q.best_ask) &
                    q.ask_size_0.ge(0) & q.bid_size_0.ge(0) &
                    (q.best_bid-q.bid_price_0).abs().le(1e-9) &
                    (q.best_ask-q.ask_price_0).abs().le(1e-9) &
                    ~q.duplicate_timestamp & ~q.non_monotonic_timestamp)
                rows += len(q)
                lo = q.timestamp_utc.min() if lo is None else min(lo, q.timestamp_utc.min())
                hi = q.timestamp_utc.max() if hi is None else max(hi, q.timestamp_utc.max())
                slugs.update(q.market_slug)
                table = pa.Table.from_pandas(q, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(destination.with_suffix('.tmp'), table.schema, compression='zstd')
                writer.write_table(table)
            inventory.append({'file': path.name, 'btc_5m_token_rows': rows, 'btc_5m_markets': len(slugs),
                              'start_utc': str(lo), 'end_utc': str(hi)})
            print(f'Secondary {path.name}: {rows} BTC 5m token snapshots', flush=True)
    finally:
        if writer:
            writer.close()
    if writer:
        destination.with_suffix('.tmp').replace(destination)
    return {'partitions': inventory, 'suspect_tick_markets': sorted(suspect),
            'quote_count': sum(p['btc_5m_token_rows'] for p in inventory)}


def select_secondary_quotes(decisions, tape_path, max_delay_ms):
    """First forward snapshot per side; never carry a quote from before the signal."""
    candidates = {'up': [], 'down': []}
    ordered = decisions.sort_values('decision_available_at')
    wanted = set(ordered.condition_id)
    columns = ['condition_id', 'timestamp_utc', 'token_side', 'best_bid', 'best_ask',
               'ask_size_0', 'quote_valid']
    for batch in pq.ParquetFile(tape_path).iter_batches(batch_size=100000, columns=columns):
        q = batch.to_pandas()
        q = q[q.condition_id.isin(wanted)]
        for side in candidates:
            token = q[q.token_side.eq(side)].sort_values('timestamp_utc')
            if token.empty:
                continue
            local = ordered[ordered.condition_id.isin(token.condition_id.unique())]
            joined = pd.merge_asof(local[['decision_id', 'condition_id', 'decision_available_at']], token,
                by='condition_id', left_on='decision_available_at', right_on='timestamp_utc',
                direction='forward', tolerance=pd.Timedelta(milliseconds=max_delay_ms))
            candidates[side].append(joined[joined.timestamp_utc.notna()])
    result = decisions.copy()
    for side, parts in candidates.items():
        if parts:
            first = pd.concat(parts).sort_values('timestamp_utc').drop_duplicates('decision_id')
            fields = ['timestamp_utc', 'best_bid', 'best_ask', 'ask_size_0', 'quote_valid']
            first = first[['decision_id'] + fields].rename(columns={c: side + '_' +
                ('ask_size' if c == 'ask_size_0' else c) for c in fields})
            result = result.merge(first, on='decision_id', how='left', validate='one_to_one')
        else:
            result[side + '_timestamp_utc'] = pd.Series(pd.NaT, index=result.index, dtype='datetime64[ns, UTC]')
            for field in ['best_bid', 'best_ask', 'ask_size']:
                result[side + '_' + field] = np.nan
            result[side + '_quote_valid'] = False
    both = result.up_timestamp_utc.notna() & result.down_timestamp_utc.notna()
    result['timestamp_utc'] = result[['up_timestamp_utc', 'down_timestamp_utc']].max(axis=1).where(both)
    result['quote_valid'] = both & result.up_quote_valid.fillna(False) & result.down_quote_valid.fillna(False)
    result['quote_delay_ms'] = (result.timestamp_utc-result.decision_available_at).dt.total_seconds()*1000
    result['side_capture_gap_ms'] = (result.up_timestamp_utc-result.down_timestamp_utc).abs().dt.total_seconds()*1000
    result['seconds_to_expiry'] = (result.market_end_utc-result.timestamp_utc).dt.total_seconds()
    result['best_prices_scope'] = 'stored_levels_only_full_book_unknown'
    return result


def reconcile_sources(primary, secondary):
    """Audit identities before union; prefer Kacho by predeclared source priority."""
    fields = ['condition_id', 'up_token_id', 'down_token_id', 'market_start_utc',
              'market_end_utc', 'polymarket_outcome_up', 'resolved_at_utc']
    common = primary[fields].merge(secondary[fields], on='condition_id', suffixes=('_primary', '_secondary'))
    mismatches = {}
    for field in fields[1:]:
        a, b = common[field + '_primary'], common[field + '_secondary']
        mismatches[field] = int((~(a.eq(b) | (a.isna() & b.isna()))).sum())
    start_collisions = primary[['condition_id', 'market_start_utc']].merge(
        secondary[['condition_id', 'market_start_utc']], on='market_start_utc', suffixes=('_primary', '_secondary'))
    mismatches['condition_id_at_same_start'] = int(start_collisions.condition_id_primary.ne(start_collisions.condition_id_secondary).sum())
    if any(mismatches.values()):
        raise ValueError(f'Conflicting source identities/settlement: {mismatches}')
    return {'common_markets': len(common), 'mismatches': mismatches,
            'selection_rule': 'First eligible Kacho decision, otherwise eligible Obadiaha; independent of PnL. One decision per market/latency.'}


def combine_decisions(primary, secondary):
    combined = pd.concat([primary, secondary], ignore_index=True)
    priority = {PRIMARY: 0, SECONDARY: 1}
    combined['_priority'] = combined.source.map(priority)
    return combined.sort_values(['eligible', '_priority'], ascending=[False, True]).drop_duplicates(
        'condition_id', keep='first').drop(columns='_priority').sort_values('decision_available_at')
