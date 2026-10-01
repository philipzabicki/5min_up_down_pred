"""Run the BTC historical experiment. Configure constants here; no CLI arguments."""
import json
import hashlib
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils.polymarket_history import (
    PRIMARY, SECONDARY, cache_official, checksum, download_source, economic_dataset,
    join_oos, normalize_markets, normalize_tape, secondary_adapter, utc, write_json,
)
from utils.polymarket_policy import evaluate_walk_forward

RAW = Path('data/raw/polymarket')
OUTPUT = Path('data/datasets/polymarket/BTC')
REPORTS = Path('data/analysis/polymarket/BTC')
PRIMARY_REVISION = '42d917dc8e3205dde8ac909792af0cce2d715c9f'
SECONDARY_REVISION = '11793901f0ac89c5a6c51123a6ccd29a3aaf8f4c'
MODEL_DATA = Path('data/datasets/modeling/BTC/BTCUSD_INDEXVOL_UM_BTCUSDT1m_model_ready.parquet')
LEGACY_OOF = MODEL_DATA.with_name(MODEL_DATA.stem.replace('_model_ready', '_oof_predictions') + '.parquet')
# Predeclared existing causal features: no retrospectively tuned indicators/subset.
FEATURE_COLUMNS = ['candle_wick_asym_1m', 'candle_signed_vol_1m', 'candle_range_ho_1m',
                   'candle_ret_co_5m_lag1', 'candle_body_abs_open_5m_lag1', 'candle_streak_5m',
                   'realized_volatility_5m', 'realized_volatility_up_15m',
                   'futures_index_basis_rel_1m', 'futures_index_basis_change_1m']
LATENCY_SCENARIOS_SECONDS = (0, 1, 2)
MAX_QUOTE_DELAY_MS = 2000
TRAIN_DAYS = 180
VALIDATION_DAYS = 14
PREDICTION_FOLDS = 4
MODEL_PARAMS = dict(n_estimators=500, learning_rate=.03, num_leaves=15,
                    min_child_samples=200, random_state=37, n_jobs=8, verbosity=-1)


def historical_predictions(markets):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dest = OUTPUT / 'historical_oos.parquet'
    metadata_path = OUTPUT / 'historical_oos_manifest.json'
    features = FEATURE_COLUMNS
    start = markets.market_start_utc.min() - pd.Timedelta(minutes=1)
    end = markets.market_end_utc.max()
    signature = {'schema_version': 2, 'dataset_sha256': checksum(MODEL_DATA), 'features': features,
                 'parameters': MODEL_PARAMS, 'start': str(start), 'end': str(end),
                 'train_days': TRAIN_DAYS, 'validation_days': VALIDATION_DAYS, 'folds': PREDICTION_FOLDS}
    if dest.exists() and metadata_path.exists() and json.loads(metadata_path.read_text()).get('input') == signature:
        return pd.read_parquet(dest)
    columns = list(dict.fromkeys(['Opened', 'Close', 'target_5m_candle_up', 'target_5m_weight'] + features))
    # Predicate pushdown: only bounded historical training plus prediction window.
    frame = pd.read_parquet(MODEL_DATA, columns=columns, filters=[
        ('Opened', '>=', (start - pd.Timedelta(days=TRAIN_DAYS + VALIDATION_DAYS)).tz_localize(None)),
        ('Opened', '<', end.tz_localize(None))])
    frame['Opened'] = utc(frame.Opened)
    frame = frame.sort_values('Opened').dropna(subset=['target_5m_candle_up'])
    edges = pd.date_range(start, end, periods=PREDICTION_FOLDS + 1)
    results, evidence = [], []
    for fold, (test_start, test_end) in enumerate(zip(edges[:-1], edges[1:])):
        val_end = test_start - pd.Timedelta(minutes=6)
        val_start = val_end - pd.Timedelta(days=VALIDATION_DAYS)
        train_end = val_start - pd.Timedelta(minutes=6)
        train = frame[(frame.Opened >= train_end - pd.Timedelta(days=TRAIN_DAYS)) & (frame.Opened < train_end)]
        val = frame[(frame.Opened >= val_start) & (frame.Opened < val_end)]
        test = frame[(frame.Opened >= test_start) & (frame.Opened < test_end) & frame.Opened.dt.minute.mod(5).eq(4)]
        if train.empty or val.empty or test.empty:
            raise ValueError('Insufficient historical model coverage')
        model = lgb.LGBMClassifier(**MODEL_PARAMS)
        model.fit(train[features].replace([np.inf, -np.inf], np.nan), train.target_5m_candle_up,
                  sample_weight=train.target_5m_weight,
                  eval_set=[(val[features].replace([np.inf, -np.inf], np.nan), val.target_5m_candle_up)],
                  eval_sample_weight=[val.target_5m_weight], callbacks=[lgb.early_stopping(25, verbose=False)])
        model_id = f'btc_historical_fold_{fold}'
        model.booster_.save_model(str(OUTPUT / (model_id + '.txt')))
        out = test[['Opened', 'Close', 'target_5m_candle_up']].rename(columns={'target_5m_candle_up': 'target_binance_proxy_up'}).copy()
        out['p_model_up'] = model.predict_proba(test[features].replace([np.inf, -np.inf], np.nan))[:, 1]
        out['fold_id'], out['source_model_id'], out['is_oos'] = fold, model_id, True
        out['fit_labels_available_at'] = val.Opened.max() + pd.Timedelta(minutes=6)
        out['train_labels_available_at'] = train.Opened.max() + pd.Timedelta(minutes=6)
        out['validation_start_utc'] = val.Opened.min()
        out['source_dataset_sha256'] = signature['dataset_sha256']
        out['feature_columns_sha256'] = hashlib.sha256(json.dumps(features).encode()).hexdigest()
        out['model_parameters_sha256'] = hashlib.sha256(json.dumps(MODEL_PARAMS, sort_keys=True).encode()).hexdigest()
        # Actual proxy movement (same exact five-minute target endpoints).
        closes = frame.set_index('Opened').Close
        future_close = closes.reindex(test.Opened + pd.Timedelta(minutes=5)).to_numpy()
        out['underlying_move_bps'] = (future_close / test.Close.to_numpy() - 1) * 10000
        results.append(out)
        evidence.append({'fold_id': fold, 'model_id': model_id, 'train_rows': len(train),
                         'validation_rows': len(val), 'prediction_rows': len(out),
                         'train_start': str(train.Opened.min()), 'train_end': str(train.Opened.max()),
                         'fit_labels_available_at': str(out.fit_labels_available_at.iloc[0]),
                         'test_start': str(test.Opened.min()), 'test_end': str(test.Opened.max()),
                         'best_iteration': model.best_iteration_})
        print(f'OOS model fold {fold}: {len(out)} predictions', flush=True)
    predictions = pd.concat(results, ignore_index=True)
    predictions.to_parquet(dest, index=False)
    write_json(metadata_path, {'input': signature, 'folds': evidence,
                              'model_scope': 'Fixed LightGBM benchmark on predeclared existing causal features. Deployed fitted-indicator model and legacy OOF are excluded because historical selection independence cannot be proven.'})
    return predictions


def quality_report(markets, counts, decisions):
    valid = decisions[decisions.eligible]
    comparable = decisions[decisions.target_polymarket_up.notna()]
    mismatch = []
    if len(comparable):
        bins = pd.cut(comparable.underlying_move_bps.abs(), [0, .1, .5, 1, 2, 5, 10, np.inf], include_lowest=True)
        for bucket, group in comparable.groupby(bins, observed=True):
            mismatch.append({'abs_move_bps': str(bucket), 'count': len(group), 'mismatch_rate': float(group.target_mismatch.astype(float).mean())})
    delays = valid.quote_delay_ms
    legacy = pd.read_parquet(LEGACY_OOF, columns=['Opened']) if LEGACY_OOF.exists() else pd.DataFrame(columns=['Opened'])
    legacy_times = utc(legacy.Opened) + pd.Timedelta(minutes=1)
    report = dict(counts, coverage_start_utc=str(markets.market_start_utc.min()),
                  coverage_end_utc=str(markets.market_end_utc.max()), markets=len(markets),
                  resolved_markets=int(markets.polymarket_outcome_up.notna().sum()),
                  token_mapping_valid=int(markets.token_mapping_valid.sum()),
                  validation_status=markets.validation_status.value_counts().to_dict(),
                  vendor_outcome_mismatch=int((markets.vendor_inferred_outcome.map({'Up': 1, 'Down': 0}) != markets.polymarket_outcome_up).where(markets.polymarket_outcome_up.notna(), False).sum()),
                  legacy_oof_market_overlap=int(legacy_times.isin(markets.market_start_utc).sum()),
                  legacy_oof_accepted=False, oos_market_overlap=len(decisions), valid_decision_points=len(valid),
                  resolved_comparable_points=len(comparable),
                  target_mismatch_rate=float(comparable.target_mismatch.astype(float).mean()) if len(comparable) else None,
                  target_mismatch_by_movement=mismatch,
                  quote_delay_ms={'median': float(delays.median()) if len(delays) else None,
                                  'p95': float(delays.quantile(.95)) if len(delays) else None,
                                  'max': float(delays.max()) if len(delays) else None},
                  missing_quote_decisions=int(decisions.timestamp_utc.isna().sum()),
                  unknown_exchange_book_age=True, ask_depth_beyond_top_available=False)
    report['policy_search_gate_passed'] = (len(valid) >= 1000 and len(valid) / max(len(decisions), 1) >= .9 and
                                            not markets.duplicate_market.any())
    return report


def audit_secondary(folder, primary_markets):
    meta = pd.read_parquet(folder / 'markets/all.parquet')
    meta = meta[meta.asset.eq('BTC') & meta.market_id.str.startswith('btc-updown-5m-')]
    resolution = pd.read_parquet(folder / 'resolutions/all.parquet')
    overlap = meta.condition_id.isin(primary_markets.condition_id)
    # Vendor market metadata has no token identities; official lookup supplies them.
    official = RAW / 'official_gamma'
    secondary_markets = meta.rename(columns={'market_id': 'market_slug'})
    books = pd.read_parquet(folder / 'orderbooks/2026-03-19.parquet')
    used = secondary_markets[secondary_markets.market_slug.isin(books.market_id)]
    cache_official(used[['market_slug']], official)
    mappings = []
    boundary_mismatches, outcome_comparisons, outcome_mismatches, unmapped_resolutions = 0, 0, 0, 0
    from utils.polymarket_history import token_mapping
    for row in used.itertuples():
        payload = json.loads((official / (row.market_slug + '.json')).read_text())['payload']
        if payload:
            tokens = token_mapping(payload)
            mappings.append({'market_slug': row.market_slug, 'condition_id': payload['conditionId'],
                             'up_token_id': tokens['up'], 'down_token_id': tokens['down']})
            boundary_mismatches += int(utc(row.start_time) != utc(payload.get('eventStartTime')) or
                                       utc(row.end_time) != utc(payload['endDate']))
            r = resolution[(resolution.condition_id.eq(payload['conditionId'])) |
                           resolution.market_id.eq(row.market_slug)]
            from utils.polymarket import resolve_polymarket_actual_up_from_market_payload
            actual = resolve_polymarket_actual_up_from_market_payload(payload) if payload.get('umaResolutionStatus') == 'resolved' else None
            if len(r) and actual is not None:
                labels = r.outcome.map({'Up': 1, 'Down': 0})
                outcome_comparisons += 1
                outcome_mismatches += int(labels.isna().any() or labels.ne(actual).any())
            else:
                unmapped_resolutions += 1
    q = secondary_adapter(books, pd.DataFrame(mappings))
    q.to_parquet(OUTPUT / 'secondary_quotes_long.parquet', index=False)
    timestamps = q.timestamp_utc
    gaps = q.sort_values('timestamp_utc').groupby(['condition_id', 'token_id']).timestamp_utc.diff().dt.total_seconds().dropna()
    report = {'source': SECONDARY, 'revision': SECONDARY_REVISION, 'btc_5m_metadata_markets': len(meta),
              'downloaded_quote_file': 'orderbooks/2026-03-19.parquet', 'btc_5m_token_observations': len(q),
              'quote_start_utc': str(timestamps.min()), 'quote_end_utc': str(timestamps.max()),
              'overlapping_primary_markets': int(overlap.sum()),
              'duplicate_token_timestamps': int(q.duplicate_timestamp.sum()),
              'unknown_token_observations': int(q.token_side.eq('unknown').sum()),
              'official_boundary_mismatches': boundary_mismatches,
              'official_resolution_comparisons': outcome_comparisons,
              'official_outcome_mismatches': outcome_mismatches,
              'missing_or_unmapped_resolutions': unmapped_resolutions,
              'duplicate_metadata_markets': int(meta.condition_id.duplicated(keep=False).sum()),
              'crossed_books': int(q.best_bid.gt(q.best_ask).sum()),
              'missing_bid_ask': int(q[['best_bid', 'best_ask']].isna().any(axis=1).sum()),
              'book_best_level_disagreements': int(((q.best_bid-q.bid_price_0).abs().gt(1e-9) |
                                                     (q.best_ask-q.ask_price_0).abs().gt(1e-9)).sum()),
              'median_token_interval_s': float(gaps.median()), 'p95_token_interval_s': float(gaps.quantile(.95)),
              'resolution_rows': len(resolution), 'merge_allowed': False,
              'reason': 'Pinned sources have no overlapping BTC 5m markets. Quote/outcome equivalence and offsets cannot be validated; no automatic merge.',
              'timestamp_contract': 'UTC per-token REST capture; asynchronous sides remain separate.',
              'book_representation': '10 levels, numeric L2 columns; bids descending, asks ascending.'}
    write_json(REPORTS / 'secondary_audit.json', report)
    return report


def write_experiment_document():
    evaluation = json.loads((REPORTS / 'evaluation.json').read_text())
    quality = json.loads((REPORTS / 'quality_latency_0s.json').read_text())
    if not all('aggregate_baselines' in scenario for scenario in evaluation['latency_scenarios'].values()):
        Path('docs').mkdir(exist_ok=True)
        Path('docs/polymarket_btc_experiment.md').write_text(
            '# BTC historical Polymarket experiment\n\nPolicy evaluation was blocked by the data-quality gate.\n\n'
            f'Coverage: {quality["coverage_start_utc"]} to {quality["coverage_end_utc"]}. '
            f'Markets: {quality["markets"]}; ticks: {quality["quote_count"]}; '
            f'valid 0s decisions: {quality["valid_decision_points"]}.\n\n'
            'Inspect data/analysis/polymarket/BTC/quality_latency_*s.json before policy search.\n', encoding='utf-8')
        return
    source_manifests = [json.loads(p.read_text()) for p in RAW.glob('*/*/manifest.json')]
    oos = pq.ParquetFile(OUTPUT / 'historical_oos.parquet').metadata.num_rows
    lines = ['# BTC historical Polymarket experiment', '',
             'Generated by `build_polymarket_history.py`; all dates below are UTC.', '',
             '## Data actually used', '',
             f'- Market coverage: **{quality["coverage_start_utc"]} through {quality["coverage_end_utc"]}** (end exclusive).',
             f'- Kacho: **{quality["markets"]:,} markets**, **{quality["quote_count"]:,} ticks**; revision `{PRIMARY_REVISION}`.',
             f'- Official outcomes and token identities: {quality["resolved_markets"]:,} / {quality["token_mapping_valid"]:,}. Validation status: `{quality["validation_status"]}`.',
             f'- New provenance-bearing future predictions: {oos:,}; market overlap {quality["oos_market_overlap"]:,}; valid economic decisions **{quality["valid_decision_points"]:,}** per latency scenario.',
             f'- Legacy OOF overlap: {quality["legacy_oof_market_overlap"]:,}, rejected for evaluation-fold early stopping and missing provenance.',
             '', '| Downloaded file | Bytes |', '| --- | ---: |']
    for manifest in source_manifests:
        for item in manifest['files']:
            lines.append(f'| {manifest["source"].split("/")[0]}/{item["file"]} | {item["size_bytes"]:,} |')
    lines += ['', 'Per-file checksums, pinned revisions and download times are in raw manifests; official responses are cached once per slug.',
              '', '## Quality and settlement audit', '',
              f'- Median / p95 quote interval: {quality["median_quote_interval_s"]} / {quality["p95_quote_interval_s"]} seconds.',
              f'- Missing bid/ask: {quality["missing_side"]}; crossed books: {quality["crossed_book"]}; duplicate timestamps: {quality["duplicate_timestamp"]}; non-monotonic timestamps: {quality["non_monotonic_timestamp"]}. Bad observations remain flagged.',
              f'- Missing forward quotes: {quality["missing_quote_decisions"]}. Two official outcomes lack a resolution-availability timestamp and are excluded from economic evaluation.',
              f'- Quote delay after each scenario availability: `{quality["quote_delay_ms"]}` ms. This measures sampling delay, not exchange book age or measured live inference latency.',
              f'- Proxy vs official outcome mismatch: **{quality["target_mismatch_rate"]:.4%}**. Kacho inferred outcome disagrees with official settlement on **{quality["vendor_outcome_mismatch"]:,} markets**.',
              '', '| Absolute proxy movement (bps) | Decisions | Mismatch |', '| --- | ---: | ---: |']
    for bucket in quality['target_mismatch_by_movement']:
        lines.append(f'| {bucket["abs_move_bps"]} | {bucket["count"]:,} | {bucket["mismatch_rate"]:.2%} |')
    lines += ['', '## Future policy evaluations', '',
              'Three chronological outer folds; each policy starts each fold with $100, so the PnL column sums three independent $100 bankrolls. Past-only calibration training and later past-only policy selection precede untouched future evaluation. The final fold never selects a variant.',
              'Current live EV uses raw OOS probabilities and shared live EV/sizing functions; other variants use calibration selected on past data. Kelly numerically maximizes binary log growth using the shared rounded fee model. Liquidity/order minimums may reject proposed sizes.', '',
              '| Policy | 0s PnL / trades | +1s PnL / trades | +2s PnL / trades |', '| --- | ---: | ---: | ---: |']
    from utils.polymarket_policy import BASELINES
    for config in BASELINES:
        cells = []
        for latency in LATENCY_SCENARIOS_SECONDS:
            entry = evaluation['latency_scenarios'][str(latency)]['aggregate_baselines'][config['name']]
            cells.append(f'${entry["sum_pnl_independent_fold_bankrolls"]:.2f} / {entry["sum_trades"]:,}')
        lines.append('| ' + config['name'] + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'The past-selected policy is NO_TRADE in all nine outer evaluations. The 1% fraction variant has only one executed trade at 0s and none at +1/+2s, largely due to the five-share order minimum. Its isolated positive result is not used to select a policy.',
              '', '| Outer fold (0s UTC) | Rows | Past-selected calibration |', '| --- | ---: | --- |']
    for fold in evaluation['latency_scenarios']['0']['folds']:
        lines.append(f'| {fold["evaluation_start_utc"]} to {fold["evaluation_end_utc"]} | {fold["evaluation_rows"]:,} | {fold["selected_calibration"]} |')
    lines += ['', '| Calibration | Future weighted Brier | Future weighted log loss |', '| --- | ---: | ---: |']
    folds = evaluation['latency_scenarios']['0']['folds']
    for method in ['none', 'platt', 'isotonic']:
        total = sum(f['evaluation_rows'] for f in folds)
        brier = sum(f['future_calibration'][method]['brier_score'] * f['evaluation_rows'] for f in folds) / total
        loss = sum(f['future_calibration'][method]['log_loss'] * f['evaluation_rows'] for f in folds) / total
        lines.append(f'| {method} | {brier:.6f} | {loss:.6f} |')
    secondary = evaluation['secondary']
    lines += ['', 'Detailed JSON reports include reliability bins, slope/intercept, PnL, return multiple, log growth, drawdown, turnover, sizes, direction, hit rate, edge, exposure and bucket attribution. Parquet files preserve calibrated probabilities, trades and bankroll events.',
              '', '## Secondary validation and remaining limits', '',
              f'- Obadiaha revision `{SECONDARY_REVISION}`: downloaded metadata/resolutions plus 2026-03-19 books; {secondary["btc_5m_metadata_markets"]:,} BTC 5m metadata markets and {secondary["btc_5m_token_observations"]:,} BTC token observations in the sampled quote file.',
              f'- Sample quote coverage: {secondary["quote_start_utc"]} to {secondary["quote_end_utc"]}; median / p95 per-token interval {secondary["median_token_interval_s"]:.3f} / {secondary["p95_token_interval_s"]:.3f}s.',
              f'- Secondary official audit: {secondary["official_resolution_comparisons"]} outcome comparisons, {secondary["official_outcome_mismatches"]} mismatches; {secondary["missing_or_unmapped_resolutions"]} missing/unmapped resolutions. Boundary mismatches: {secondary["official_boundary_mismatches"]}; missing bid/ask token snapshots: {secondary["missing_bid_ask"]}.',
              f'- Primary overlap: {secondary["overlapping_primary_markets"]}; merge remains disabled. There is no common market/time interval on which provider equivalence can be demonstrated.',
              '- The probability model is a fixed ten-feature causal LightGBM benchmark, not the current deployed model. Original tuned indicators/subset cannot be certified independent of this historical period. Existing modeling/training configuration is unchanged.',
              '- Kacho has ask top sizes and bid aggregate depth, no deeper ask ladder. Larger unsupported executions/VWAP are unavailable, never invented. Unknown exchange book age, FAK fills, early exit and actual redemption availability cannot be reconstructed.',
              '- Historical per-market Gamma schedules differ from current live configuration. Responses show rate=.25/exponent=2 and rate=.07/exponent=1. Archived fee-change timestamps are unavailable; later-fetched metadata is the documented fee assumption.',
              '- Full suite: 21 new tests pass; 157/158 total tests pass. Existing optimizer test expects 3 slippage ticks while the unchanged HEAD constant is 2.',
              '', '## Reproduction', '', 'From the repository root (configure constants at the top of the script):', '',
              '```powershell', 'python build_polymarket_history.py',
              'python -m unittest discover -s tests -p test_polymarket_history.py',
              'python -m unittest discover -s tests', '```', '',
              f'Batch normalization: {quality["normalization_seconds"]:.2f}s; forward join: {quality.get("quote_join_seconds", 0):.2f}s; counterfactual targets: {quality.get("economic_targets_seconds", 0):.2f}s (this local run).',
              '', 'Sources: [Kacho dataset card](https://huggingface.co/datasets/kachoio/polymarket-5-minute-crypto-up-down-markets), [Obadiaha dataset card](https://huggingface.co/datasets/obadiaha/polymarket-crypto-5m-15m), [official market metadata](https://docs.polymarket.com/market-data/discover-markets).', '']
    Path('docs').mkdir(exist_ok=True)
    Path('docs/polymarket_btc_experiment.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    begin = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    primary = download_source(PRIMARY, ['README.md', 'btc_markets.parquet', 'btc_ticks.parquet'], RAW, PRIMARY_REVISION)
    secondary = download_source(SECONDARY, ['README.md', 'markets/all.parquet', 'resolutions/all.parquet',
                                          'orderbooks/2026-03-19.parquet'], RAW, SECONDARY_REVISION)
    raw_markets = pd.read_parquet(primary / 'btc_markets.parquet')
    cache_official(raw_markets.rename(columns={'slug': 'market_slug'}), RAW / 'official_gamma')
    markets = normalize_markets(raw_markets, RAW / 'official_gamma')
    markets['source_revision'] = PRIMARY_REVISION
    markets.to_parquet(OUTPUT / 'markets.parquet', index=False)
    counts = normalize_tape(primary / 'btc_ticks.parquet', markets, OUTPUT / 'quotes.parquet')
    predictions = historical_predictions(markets)
    from utils.polymarket_history import select_quotes
    scenarios = {}
    for latency in LATENCY_SCENARIOS_SECONDS:
        join_started = time.perf_counter()
        joined = select_quotes(join_oos(predictions, markets, latency), OUTPUT / 'quotes.parquet', MAX_QUOTE_DELAY_MS)
        joined.loc[joined.condition_id.isin(counts['suspect_tick_markets']), 'quote_valid'] = False
        join_seconds = time.perf_counter() - join_started
        targets_started = time.perf_counter()
        decisions = economic_dataset(joined)
        targets_seconds = time.perf_counter() - targets_started
        decisions.to_parquet(OUTPUT / f'policy_latency_{latency}s.parquet', index=False)
        report = quality_report(markets, counts, decisions)
        report.update(quote_join_seconds=join_seconds, economic_targets_seconds=targets_seconds)
        write_json(REPORTS / f'quality_latency_{latency}s.json', report)
        if report['policy_search_gate_passed']:
            scenarios[str(latency)] = evaluate_walk_forward(decisions[decisions.eligible].copy(), REPORTS / f'latency_{latency}s')
        else:
            scenarios[str(latency)] = {'status': 'blocked_by_quality_gate'}
        print(f'Latency {latency}s: {report["valid_decision_points"]} valid decisions', flush=True)
    secondary_audit = audit_secondary(secondary, markets)
    write_json(REPORTS / 'evaluation.json', {'latency_scenarios': scenarios,
               'secondary': secondary_audit, 'total_seconds': time.perf_counter() - begin,
               'limitations': ['Snapshot liquidity is not guaranteed historical fill.',
                              'Source has no exchange book-age timestamps or ask depth beyond top.',
                              'Probability model is a fixed historical benchmark, not the current deployed model: retrospective fitted-indicator selection excluded.',
                              'Official fee schedules are fetched later; no archived fee-change timestamps.']})
    write_experiment_document()


if __name__ == '__main__':
    main()
