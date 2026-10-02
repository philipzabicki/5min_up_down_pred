"""Run the BTC historical experiment. Configure constants here; no CLI arguments."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from requests import RequestException

from utils.data import resolve_oof_prediction_output_paths
from utils.project_config import load_modeling_settings, load_runtime_artifact_paths
from utils.polymarket_history import (
    PRIMARY, SECONDARY, cache_official, checksum, combine_decisions, download_source,
    economic_dataset, join_oos, normalize_markets, normalize_secondary_tapes,
    normalize_tape, reconcile_sources, select_quotes, select_secondary_quotes,
    session, token_mapping, utc, validate_oos, write_json,
)
from utils.polymarket_policy import evaluate_walk_forward

RAW = Path('data/raw/polymarket')
OUTPUT = Path('data/datasets/polymarket/BTC')
REPORTS = Path('data/analysis/polymarket/BTC')
LATENCY_SCENARIOS_SECONDS = (0, 1, 2)
MAX_QUOTE_DELAY_MS = 2000
OOF_PROBABILITY_COLUMN = 'oof_pred_proba_up'
SCHEMA_VERSION = 3


def ranges(times, step=pd.Timedelta(minutes=5)):
    """Compact actual coverage/gaps without filling absent observations."""
    times = pd.Series(utc(times)).dropna().drop_duplicates().sort_values()
    if times.empty:
        return []
    groups = times.diff().ne(step).cumsum()
    return [{'start_utc': str(g.min()), 'last_utc': str(g.max()), 'count': len(g)}
            for _, g in times.groupby(groups)]


def load_current_oof():
    settings = load_modeling_settings(asset='BTC')
    path = resolve_oof_prediction_output_paths(settings, preview_rows=1000)['parquet']
    if not path.exists():
        raise FileNotFoundError(f'Current BTC OOF artifact missing: {path.resolve()}')
    prediction = pd.read_parquet(path, columns=['Opened', 'Close', 'target_5m_candle_up', OOF_PROBABILITY_COLUMN])
    prediction = prediction.rename(columns={OOF_PROBABILITY_COLUMN: 'p_model_up',
                                           'target_5m_candle_up': 'target_binance_proxy_up'})
    prediction['Opened'] = utc(prediction.Opened)
    validate_oos(prediction)
    metadata_path = load_runtime_artifact_paths(asset='BTC')['model_meta_path']
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    declared = metadata.get('oof_predictions', {})
    metadata_matches = (declared.get('rows') == len(prediction) and
                        declared.get('prediction_col') == OOF_PROBABILITY_COLUMN and
                        Path(declared.get('path', '')).resolve() == path.resolve())
    if not metadata_matches:
        raise ValueError(f'Current runtime model metadata does not match OOF: {metadata_path}')
    # Future closes are used only for target agreement diagnostics.
    closes = prediction.set_index('Opened').Close
    future_close = closes.reindex(prediction.Opened + pd.Timedelta(minutes=5)).to_numpy()
    prediction['underlying_move_bps'] = (future_close / prediction.Close.to_numpy() - 1) * 10000
    decisions = prediction[prediction.Opened.dt.minute.mod(5).eq(4)].copy()
    expected = pd.date_range(decisions.Opened.min(), decisions.Opened.max(), freq='5min')
    missing = expected.difference(pd.DatetimeIndex(decisions.Opened))
    manifest = {'path': str(path.resolve()), 'sha256': checksum(path),
                'modified_at_utc': str(pd.Timestamp(path.stat().st_mtime, unit='s', tz='UTC')),
                'size_bytes': path.stat().st_size, 'rows': len(prediction),
                'probability_column': OOF_PROBABILITY_COLUMN,
                'probability_min': float(prediction.p_model_up.min()),
                'probability_max': float(prediction.p_model_up.max()),
                'start_utc': str(prediction.Opened.min()), 'end_utc': str(prediction.Opened.max()),
                'decision_rows': len(decisions), 'decision_grid_coverage': len(decisions) / len(expected),
                'missing_decision_rows': len(missing), 'missing_decision_ranges': ranges(missing),
                'metadata_path': str(metadata_path.resolve()), 'metadata_sha256': checksum(metadata_path),
                'metadata_oof': declared, 'metadata_matches': metadata_matches,
                'training_assumption': 'User accepts main-model OOF construction including evaluation-fold early stopping. No retraining, no final-model predictions, no benchmark fallback.',
                'timestamp_contract': 'Opened is the UTC opening of a closed 1m candle. Probability available at Opened+1m+scenario latency; minute%5==4 only.'}
    return decisions, manifest


def experiment_signature(oof_manifest, inventories):
    inputs = {'schema_version': SCHEMA_VERSION, 'oof_sha256': oof_manifest['sha256'],
              'metadata_sha256': oof_manifest['metadata_sha256'],
              'sources': {source: data['sha'] for source, data in inventories.items()},
              'latencies': LATENCY_SCENARIOS_SECONDS, 'max_quote_delay_ms': MAX_QUOTE_DELAY_MS,
              'code_and_config': {name: checksum(name) for name in [
                  'build_polymarket_history.py', 'utils/polymarket_history.py', 'utils/polymarket_policy.py',
                  'utils/trading.py', 'utils/polymarket.py', 'utils/project_config.py',
                  'configs/runtime/active.json', 'configs/runtime/trade_policy_project.json', 'configs/live.json']}}
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(), inputs


def secondary_markets(folder):
    meta = pd.read_parquet(folder / 'markets/all.parquet')
    meta = meta[meta.asset.eq('BTC') & meta.market_id.str.startswith('btc-updown-5m-')].copy()
    meta = meta.drop(columns='slug').rename(columns={'market_id': 'market_slug', 'start_time': 'market_start_utc', 'end_time': 'market_end_utc'})
    cache_official(meta[['market_slug']], RAW / 'official_gamma')
    tokens = []
    for row in meta.itertuples():
        payload = json.loads((RAW / 'official_gamma' / (row.market_slug + '.json')).read_text())['payload']
        try:
            mapping = token_mapping(payload) if payload else {}
        except (ValueError, TypeError, KeyError):
            mapping = {}
        tokens.append({'up_token_id': mapping.get('up'), 'down_token_id': mapping.get('down')})
    meta[['up_token_id', 'down_token_id']] = pd.DataFrame(tokens, index=meta.index)
    meta['vendor_inferred_outcome'] = None
    m = normalize_markets(meta, RAW / 'official_gamma')
    m['source'] = SECONDARY
    resolution = pd.read_parquet(folder / 'resolutions/all.parquet')
    resolution = resolution[resolution.asset.eq('BTC') & resolution.market_id.str.startswith('btc-updown-5m-')]
    joined = m.merge(resolution[['market_id', 'condition_id', 'outcome', 'resolved_at']],
                     left_on='market_slug', right_on='market_id', how='left', suffixes=('', '_resolution'))
    comparable = joined.polymarket_outcome_up.notna() & joined.outcome.notna()
    disagreement = comparable & (joined.outcome.map({'Up': 1, 'Down': 0}).ne(joined.polymarket_outcome_up) |
                                  joined.condition_id.ne(joined.condition_id_resolution))
    m.loc[m.condition_id.isin(joined.loc[disagreement, 'condition_id']), 'validation_status'] = 'secondary_resolution_disagreement'
    audit = {'official_resolution_comparisons': int(comparable.sum()),
             'official_outcome_or_identity_mismatches': int(disagreement.sum()),
             'missing_vendor_resolution': int(joined.outcome.isna().sum()),
             'vendor_resolution_time_missing': int(joined.resolved_at.isna().sum()),
             'settlement_time_source': 'Later official Gamma umaEndDate/closedTime; vendor times audit only.'}
    return m, audit


def coverage_report(markets, predictions, decisions, counts, inventory, folder):
    starts = predictions.Opened + pd.Timedelta(minutes=1)
    covered = markets[markets.market_start_utc.isin(starts)]
    valid = decisions[decisions.eligible]
    comparable = decisions[decisions.target_polymarket_up.notna() & decisions.target_binance_proxy_up.notna()]
    all_times = pd.date_range(markets.market_start_utc.min(), markets.market_start_utc.max(), freq='5min')
    manifest = json.loads((folder / 'manifest.json').read_text())
    return {'source': markets.source.iloc[0], 'revision': inventory['sha'],
            'available_files': [item['rfilename'] for item in inventory['siblings']],
            'downloaded_files': manifest['files'], 'available_market_ranges': ranges(markets.market_start_utc),
            'available_start_utc': str(markets.market_start_utc.min()), 'available_end_utc': str(markets.market_end_utc.max()),
            'markets': len(markets), 'downloaded_quote_coverage': manifest.get('coverage', {}),
            'btc_quote_inventory': counts, 'oof_covered_markets': len(covered),
            'oof_covered_ranges': ranges(covered.market_start_utc),
            'missing_oof_markets': len(markets)-len(covered),
            'missing_oof_ranges': ranges(markets.loc[~markets.market_start_utc.isin(starts), 'market_start_utc']),
            'missing_market_grid_ranges': ranges(all_times.difference(pd.DatetimeIndex(markets.market_start_utc))),
            'validation_status': markets.validation_status.value_counts().to_dict(),
            'quote_and_settlement_valid_markets': len(valid), 'valid_ranges': ranges(valid.market_start_utc),
            'exclusions': decisions.loc[~decisions.eligible, 'exclusion_reason'].value_counts().to_dict(),
            'exclusion_ranges': {reason: ranges(g.market_start_utc) for reason, g in
                                 decisions.loc[~decisions.eligible].groupby('exclusion_reason')},
            'target_mismatch_rate': float(comparable.target_mismatch.astype(float).mean()) if len(comparable) else None}


def write_experiment_document(evaluation):
    oof = evaluation['oof']
    lines = ['# BTC historical Polymarket experiment', '',
             'Generated by `build_polymarket_history.py`. All timestamps are UTC.', '',
             f'Current main-model OOF: `{oof["path"]}`.',
             f'SHA256: `{oof["sha256"]}`. Modified: {oof["modified_at_utc"]}.',
             f'Predictions: {oof["rows"]:,}; decision rows: {oof["decision_rows"]:,}; probability column: `{oof["probability_column"]}`.',
             f'Opened range: {oof["start_utc"]} to {oof["end_utc"]}; decision-grid coverage {oof["decision_grid_coverage"]:.6%}; missing decisions {oof["missing_decision_rows"]}.',
             f'Model metadata: `{oof["metadata_path"]}`; training OOF coverage: {oof["metadata_oof"].get("coverage_ratio")}.',
             oof['training_assumption'],
             'The ten-feature benchmark, its training/configuration and predictions are removed from the active pipeline. Existing old artifacts are retained but never read.', '',
             f'Experiment/cache fingerprint: `{evaluation["fingerprint"]}`. Probability-dependent outputs are rebuilt in its run directory. Raw data and verified official settlement caches are retained.', '',
             '## Coverage', '',
             '| Source / latency | Available markets UTC | Markets | OOF covered | Valid quotes + official settlement |',
             '| --- | --- | ---: | ---: | ---: |']
    for latency, sources in evaluation['coverage'].items():
        for name, c in sources.items():
            lines.append(f'| {name} / {latency}s | {c["available_start_utc"]} to {c["available_end_utc"]} | {c["markets"]:,} | {c["oof_covered_markets"]:,} | {c["quote_and_settlement_valid_markets"]:,} |')
    lines += ['', 'All currently listed orderbook partitions are inspected in batches, including partitions without BTC 5m rows. Exact BTC ranges, file hashes, gaps and exclusions are in `coverage.json`; no single-day extrapolation.',
              f'Source reconciliation: `{evaluation["source_reconciliation"]}`.',
              'Obadiaha uses independently captured UP/DOWN books: first forward snapshot per side within 2s of signal availability; execution at the later capture. Side timestamps/gap remain in decision tables. This is an asynchronous snapshot execution assumption, not proof of a simultaneous book or guaranteed fill.', '',
              '## Continuous economic evaluation', '',
              'The first 40% of eligible chronological markets initializes past calibration/policy selection. The remaining 60% is evaluated in three folds. Every fixed policy and the past-selected strategy has one independent $100 portfolio per latency; no resets, capital additions or aggregation across latencies.',
              'Calibration/policy selection use only labels officially settled before the fold starts. Inner calibration training precedes policy selection. Live uses raw OOF; other policies use the past-selected calibrator. The selected strategy can change policy while existing positions keep their original payout and settlement schedule.',
              'Cash pays gross stake including fees. Stake stays locked until max(official resolution, expiry); redemption delay is assumed zero after that instant. Open-position cost, cash, realized PnL and unavailable market valuation are separate columns. Drawdown uses cash + open-position cost and is **not mark-to-market**. Final balances include all settlements.', '',
              '| Policy | Latency | Initial USD | Final USD | PnL USD | Trades | Max drawdown (cost basis) |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for latency, scenario in evaluation['latency_scenarios'].items():
        if 'continuous_baselines' not in scenario:
            lines += [f'| blocked: {scenario} | {latency}s | | | | | |']
            continue
        for name, result in scenario['continuous_baselines'].items():
            lines.append(f'| {name} | {latency}s | {result["initial_bankroll"]:.2f} | {result["final_balance"]:.2f} | {result["net_pnl"]:.2f} | {result["number_trades"]:,} | {result["max_drawdown"]:.2%} |')
    lines += ['', 'Evaluation periods and walk-forward choices:']
    for latency, scenario in evaluation['latency_scenarios'].items():
        if 'folds' in scenario:
            lines += [f'- {latency}s: {scenario["evaluation_start_utc"]} to {scenario["evaluation_end_utc"]}; {scenario["evaluation_rows"]:,} markets; policies {[f["selected_policy"] for f in scenario["folds"]]}; calibrators {[f["selected_calibration"] for f in scenario["folds"]]}.']
    lines += ['', 'The combined evaluation warmup includes the entire earlier Obadiaha period. Independent source evaluations below use the same walk-forward functions so that its verified history also receives economic evaluation. Each source/policy/latency has its own continuous $100 portfolio; these results are never added to the combined portfolio.', '',
              '| Independent source | Latency | Evaluation UTC | Markets |', '| --- | ---: | --- | ---: |']
    for latency, sources in evaluation['source_evaluations'].items():
        for name, scenario in sources.items():
            if 'folds' in scenario:
                lines.append(f'| {name} | {latency}s | {scenario["evaluation_start_utc"]} to {scenario["evaluation_end_utc"]} | {scenario["evaluation_rows"]:,} |')
    lines += ['', '| Source | Policy | Latency | Initial USD | Final USD | PnL USD | Trades | Max drawdown (cost basis) |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for latency, sources in evaluation['source_evaluations'].items():
        for source, scenario in sources.items():
            for name, result in scenario.get('continuous_baselines', {}).items():
                lines.append(f'| {source} | {name} | {latency}s | {result["initial_bankroll"]:.2f} | {result["final_balance"]:.2f} | {result["net_pnl"]:.2f} | {result["number_trades"]:,} | {result["max_drawdown"]:.2%} |')
    lines += ['', '## Limits and reproduction', ''] + ['- ' + text for text in evaluation['limitations']]
    validation_path = REPORTS / 'validation.json'
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        lines += ['', f'Tests: focused {validation["focused"]["passed"]}/{validation["focused"]["run"]}; full suite {validation["full"]["passed"]}/{validation["full"]["run"]}.',
                  validation.get('preexisting_failure', '')]
    lines += ['', 'Validation details: `data/analysis/polymarket/BTC/validation.json`.', '',
              '```powershell', 'python build_polymarket_history.py',
              'python -m unittest discover -s tests -p test_polymarket_history.py',
              'python -m unittest discover -s tests', '```', '',
              'Sources: [Kacho dataset](https://huggingface.co/datasets/kachoio/polymarket-5-minute-crypto-up-down-markets), [Obadiaha dataset](https://huggingface.co/datasets/obadiaha/polymarket-crypto-5m-15m).']
    Path('docs/polymarket_btc_experiment.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    begin = time.perf_counter()
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS / 'evaluation.json', {'status': 'running', 'schema_version': SCHEMA_VERSION})
    predictions, oof = load_current_oof()
    write_json(REPORTS / 'oof_manifest.json', oof)
    inventories = {}
    client = session()
    for source in [PRIMARY, SECONDARY]:
        response = client.get('https://huggingface.co/api/datasets/' + source, timeout=60)
        response.raise_for_status()
        inventories[source] = response.json()
    fingerprint, inputs = experiment_signature(oof, inventories)
    output, reports = OUTPUT / 'runs' / fingerprint[:16], REPORTS / 'runs' / fingerprint[:16]
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / 'inputs.json', inputs)
    primary = download_source(PRIMARY, ['README.md', 'btc_markets.parquet', 'btc_ticks.parquet'], RAW, inventories[PRIMARY]['sha'])
    secondary_files = ['README.md', 'markets/all.parquet', 'resolutions/all.parquet'] + sorted(
        x['rfilename'] for x in inventories[SECONDARY]['siblings'] if x['rfilename'].startswith('orderbooks/') and x['rfilename'].endswith('.parquet'))
    secondary = download_source(SECONDARY, secondary_files, RAW, inventories[SECONDARY]['sha'])
    raw_markets = pd.read_parquet(primary / 'btc_markets.parquet')
    cache_official(raw_markets.rename(columns={'slug': 'market_slug'}), RAW / 'official_gamma')
    markets = normalize_markets(raw_markets, RAW / 'official_gamma')
    secondary_m, secondary_audit = secondary_markets(secondary)
    reconciliation = reconcile_sources(markets, secondary_m)
    for m, source in [(markets, PRIMARY), (secondary_m, SECONDARY)]:
        m['source_revision'] = inventories[source]['sha']
    markets.to_parquet(output / 'markets.parquet', index=False)
    secondary_m.to_parquet(output / 'secondary_markets.parquet', index=False)
    counts = normalize_tape(primary / 'btc_ticks.parquet', markets, output / 'quotes.parquet')
    secondary_counts = normalize_secondary_tapes([secondary / name for name in secondary_files if name.startswith('orderbooks/')],
                                               secondary_m, output / 'secondary_quotes_long.parquet')
    scenarios, coverage, source_evaluations = {}, {}, {}
    for latency in LATENCY_SCENARIOS_SECONDS:
        joined = select_quotes(join_oos(predictions, markets, latency), output / 'quotes.parquet', MAX_QUOTE_DELAY_MS)
        joined.loc[joined.condition_id.isin(counts['suspect_tick_markets']), 'quote_valid'] = False
        primary_d = economic_dataset(joined)
        secondary_joined = select_secondary_quotes(join_oos(predictions, secondary_m, latency), output / 'secondary_quotes_long.parquet', MAX_QUOTE_DELAY_MS)
        secondary_joined.loc[secondary_joined.condition_id.isin(secondary_counts['suspect_tick_markets']), 'quote_valid'] = False
        secondary_d = economic_dataset(secondary_joined)
        coverage[str(latency)] = {
            'kacho': coverage_report(markets, predictions, primary_d, counts, inventories[PRIMARY], primary),
            'obadiaha': coverage_report(secondary_m, predictions, secondary_d, secondary_counts, inventories[SECONDARY], secondary)}
        combined = combine_decisions(primary_d, secondary_d)
        combined.to_parquet(output / f'policy_latency_{latency}s.parquet', index=False)
        primary_d.to_parquet(output / f'kacho_latency_{latency}s.parquet', index=False)
        secondary_d.to_parquet(output / f'obadiaha_latency_{latency}s.parquet', index=False)
        valid = combined[combined.eligible].copy()
        if len(valid) >= 250 and not valid.condition_id.duplicated().any():
            scenarios[str(latency)] = evaluate_walk_forward(valid, reports / f'latency_{latency}s')
        else:
            scenarios[str(latency)] = {'status': 'insufficient_valid_decisions', 'rows': len(valid)}
        source_evaluations[str(latency)] = {}
        for name, decisions in [('kacho', primary_d), ('obadiaha', secondary_d)]:
            eligible = decisions[decisions.eligible].copy()
            print(f'Independent evaluation {name}, latency {latency}s: {len(eligible)} eligible markets', flush=True)
            source_evaluations[str(latency)][name] = (evaluate_walk_forward(eligible, reports / name / f'latency_{latency}s')
                if len(eligible) >= 250 else {'status': 'insufficient_valid_decisions', 'rows': len(eligible)})
        print(f'Latency {latency}s: {len(valid)} valid unique markets', flush=True)
    evaluation = {'status': 'complete', 'fingerprint': fingerprint, 'oof': oof,
                  'output_directory': str(output.resolve()), 'report_directory': str(reports.resolve()),
                  'latency_scenarios': scenarios, 'source_evaluations': source_evaluations,
                  'coverage': coverage, 'source_reconciliation': reconciliation,
                  'secondary_audit': secondary_audit, 'total_seconds': time.perf_counter()-begin,
                  'limitations': [
                      'Accepted main-model OOF includes evaluation-fold early stopping; this is a retrospective economic experiment, not a claim of prospective training independence.',
                      'Snapshot liquidity is not a guaranteed FAK fill. No deeper ask liquidity is invented; stake requires observed top ask capacity.',
                      'Kacho exchange book age is unknown; Obadiaha side snapshots are asynchronous. Latencies 0/1/2s are assumptions, not measured inference timings.',
                      'Official financial outcome comes from Gamma; Binance target is only an agreement diagnostic.',
                      'Historical official fee metadata was fetched later; fee-change timestamps are unavailable.',
                      'Live entry EV/direction and sizing reuse current runtime configuration/functions. Early exits, actual fills, redemption delays and bankroll withdrawal caps cannot be reconstructed and are not simulated.',
                      'Cash is released at official settlement availability with zero additional redemption delay. Drawdown is cost basis, not mark-to-market.',
                      'Data ends where current source partitions end; no unavailable history or missing OOF is fabricated.']}
    write_json(reports / 'evaluation.json', evaluation)
    write_json(REPORTS / 'coverage.json', coverage)
    write_json(REPORTS / 'evaluation.json', evaluation)
    write_experiment_document(evaluation)
    print(f'Experiment complete: {reports}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError, RequestException) as error:
        write_json(REPORTS / 'evaluation.json', {'status': 'blocked', 'reason': str(error)})
        Path('docs/polymarket_btc_experiment.md').write_text(
            '# BTC historical Polymarket experiment\n\nExecution blocked: ' + str(error) +
            '\n\nNo substitute model or simulated result was produced.\n', encoding='utf-8')
        raise
