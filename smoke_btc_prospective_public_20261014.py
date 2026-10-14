"""Public read-only smoke over three consecutive BTC five-minute markets."""
from __future__ import annotations

import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

import run_btc_prospective_shadow as prospective
import audit_feature_readiness as audit


ROOT = Path(__file__).resolve().parent
SMOKE_MARKET_COUNT = 3
OUT_DIR = ROOT / "data/analysis/polymarket/BTC/btc_prospective_test_20261014/smoke"


def _metadata(start):
    with requests.Session() as session:
        session.headers.update({"User-Agent": "btc-prospective-public-smoke/1.0"})
        return prospective._market_metadata(session, start)


def _prepare_runtime():
    bundle = json.loads(prospective.BUNDLE_PATH.read_text(encoding="utf-8"))
    runtime_dir = OUT_DIR / "runtime"
    requirement_path, requirements = prospective._runtime_requirements(
        json.loads((ROOT / bundle["model_meta_path"]).read_text(encoding="utf-8")),
        ROOT / bundle["indicator_fit_results_dir"],
        runtime_dir / "indicator_history_requirements.json",
    )
    runtime_window = max(int(requirements["global_required_runtime_window"]), int(audit.DEFAULT_MAX_KEEP))
    connection = prospective._connect(Path(":memory:"))
    predictor, _ = prospective._build_predictor(
        connection, bundle, requirement_path, runtime_window, runtime_dir=runtime_dir
    )
    calibrator_path = ROOT / bundle["calibrator_path"]
    calibrator = json.loads(calibrator_path.read_text(encoding="utf-8"))
    last_opened = prospective._utc(predictor.opened_candles[-1])
    latest_closed = prospective._utc().floor("min") - prospective.live_runtime.INTERVAL_DELTA
    warmup_started = time.perf_counter()
    warmup_received = None
    warmup_seconds = 0.0
    if latest_closed > last_opened:
        catchup, warmup_started_at, warmup_received, warmup_seconds = prospective._timed_fetch_catchup_pair(
            last_opened + prospective.live_runtime.INTERVAL_DELTA, latest_closed
        )
        prospective._advance_predictor(predictor, catchup)
        prospective._persist_candles(connection, catchup.itertuples(index=False), warmup_received, {
            "source": "public_rest_smoke_warmup",
            "request_started_at_utc": prospective._iso(warmup_started_at),
            "received_at_utc": prospective._iso(warmup_received),
            "request_elapsed_seconds": warmup_seconds,
        })
    return connection, predictor, predictor.model, calibrator, {
        "runtime_window_candles": runtime_window,
        "warmup_first_opened_utc": None if latest_closed <= last_opened else catchup.Opened.iloc[0].isoformat(),
        "warmup_last_opened_utc": prospective._iso(latest_closed),
        "warmup_received_at_utc": None if warmup_received is None else prospective._iso(warmup_received),
        "warmup_fetch_seconds": warmup_seconds,
        "warmup_elapsed_seconds": time.perf_counter() - warmup_started,
    }


def _capture(start, metadata, predictor, model, calibrator, connection, feed):
    deadline = start - pd.Timedelta(minutes=1)
    input_opened = deadline - pd.Timedelta(minutes=1)
    prospective._wait_until(deadline)
    started = prospective._utc()
    book_sessions = [requests.Session(), requests.Session()]
    for session in book_sessions:
        session.headers.update({"User-Agent": "btc-prospective-public-smoke/1.0"})
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            up_future = pool.submit(
                prospective.shadow._book_observation,
                book_sessions[0], metadata["tokens"]["up"],
            )
            down_future = pool.submit(
                prospective.shadow._book_observation,
                book_sessions[1], metadata["tokens"]["down"],
            )
            books, book_errors = prospective._collect_book_futures(up_future, down_future)
    finally:
        for book_session in book_sessions:
            book_session.close()
    t60 = prospective._quote_snapshot(metadata, "T-60", books=books, book_errors=book_errors)
    websocket_wait_started = time.perf_counter()
    prospective._advance_websocket_feed_until(
        connection, predictor, feed, input_opened,
        prospective._websocket_pair_wait_budget(start - pd.Timedelta(seconds=59)),
    )
    websocket_wait_seconds = time.perf_counter() - websocket_wait_started
    last_opened = prospective._utc(predictor.opened_candles[-1])
    if last_opened > input_opened:
        raise RuntimeError(f"Predictor is beyond the T-60 candle: {last_opened} > {input_opened}")
    fallback = None
    if last_opened < input_opened:
        candles, btc_started, received, btc_elapsed = prospective._timed_fetch_catchup_pair(
            last_opened + prospective.live_runtime.INTERVAL_DELTA, input_opened
        )
        prospective._advance_predictor(predictor, candles)
        prospective._persist_candles(connection, candles.itertuples(index=False), received, {
            "source": "public_rest_smoke_t60_fallback",
            "request_started_at_utc": prospective._iso(btc_started),
            "received_at_utc": prospective._iso(received),
            "request_elapsed_seconds": btc_elapsed,
        })
        fallback = {"request_started_at_utc": prospective._iso(btc_started), "request_elapsed_seconds": btc_elapsed}
    stored_input = prospective._persisted_candle(connection, input_opened)
    if stored_input is None:
        raise RuntimeError("T-60 BTC candle is absent from the smoke candle store")
    received = prospective._utc(stored_input["received_at_utc"])
    source_metadata = stored_input["source_metadata"]
    attempt = prospective._decision_attempt(predictor, model, calibrator, metadata, start, t60, received)
    attempt["timings"]["btc_source_metadata"] = source_metadata
    attempt["timings"]["btc_websocket_pair_wait_seconds"] = websocket_wait_seconds
    attempt["timings"]["btc_rest_fallback"] = fallback
    attempt["timings"]["btc_websocket_connections"] = feed.status_snapshot()
    return {
        "market_slug": metadata["slug"],
        "market_start_utc": start.isoformat(),
        "condition_id": metadata["condition_id"],
        "metadata_received_at_utc": metadata["received_at_utc"],
        "t60_deadline_utc": deadline.isoformat(),
        "t60_cycle_started_at_utc": started.isoformat(),
        "btc_input_opened_utc": input_opened.isoformat(),
        "btc_data_received_at_utc": received.isoformat(),
        "btc_source_metadata": source_metadata,
        "btc_rest_fallback": fallback,
        "btc_websocket_connections": feed.status_snapshot(),
        "t60_snapshot": attempt.get("t60_snapshot"),
        "t59_snapshot": attempt.get("t59_snapshot"),
        "probabilities": attempt.get("probabilities"),
        "policy": attempt.get("policy"),
        "attempt_status": attempt.get("attempt_status"),
        "reason": attempt.get("reason"),
        "timings": attempt.get("timings"),
        "one_second_budget_met": bool(attempt.get("timings", {}).get("quality_inputs_ready_before_t59")),
        "orders_enabled": False,
    }


def run_smoke():
    started_at = pd.Timestamp.now(tz="UTC")
    artifact_hashes = prospective._artifact_manifest()
    runtime_connection, predictor, model, calibrator, warmup = _prepare_runtime()
    feed = prospective.PublicBtcCandleFeed()
    feed.start()
    feed_connections = prospective._wait_for_public_feed_connections(feed)
    startup_backfill = prospective._backfill_predictor_to_latest_closed(
        runtime_connection, predictor, "public_rest_smoke_websocket_startup_backfill"
    )
    warmup["post_connect_startup_backfill"] = startup_backfill
    print(
        f"[public-smoke] model runtime ready; last candle={prospective._iso(predictor.opened_candles[-1])}; "
        f"warmup={warmup['warmup_elapsed_seconds']:.2f}s; "
        f"startup_backfill={json.dumps(startup_backfill, sort_keys=True)}; "
        f"feeds={json.dumps(feed_connections, sort_keys=True)}",
        flush=True,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "btc-prospective-public-smoke/1.0"})
    first_start = prospective._next_market_start(prospective._utc())
    captures = []
    errors = []
    for offset in range(SMOKE_MARKET_COUNT):
        start = first_start + pd.Timedelta(minutes=5 * offset)
        try:
            metadata = prospective._market_metadata(session, start)
            captures.append(_capture(start, metadata, predictor, model, calibrator, runtime_connection, feed))
        except Exception as exc:
            errors.append({
                "market_start_utc": start.isoformat(),
                "error": f"{type(exc).__name__}:{exc}",
                "recorded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            })
            print(f"[public-smoke] capture_error={errors[-1]['error']}", flush=True)
        print(f"[public-smoke] {start.isoformat()} captured={len(captures)} errors={len(errors)}", flush=True)

    settlement_deadline = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=10)
    settlements = {}
    while pd.Timestamp.now(tz="UTC") < settlement_deadline and len(settlements) < len(captures):
        for item in captures:
            slug = item["market_slug"]
            if slug in settlements:
                continue
            try:
                outcome = prospective._official_outcome(session, slug)
                if outcome is not None:
                    settlements[slug] = {
                        "target_up": int(outcome["target_up"]),
                        "resolved_at_utc": outcome["resolved_at_utc"],
                        "retrieved_at_utc": outcome["observed_at_utc"],
                    }
            except requests.RequestException as exc:
                errors.append({"market_slug": slug, "outcome_error": f"{type(exc).__name__}:{exc}"})
        if len(settlements) < len(captures):
            time.sleep(10)

    report = {
        "smoke_id": "btc_prospective_public_smoke_20261014",
        "started_at_utc": started_at.isoformat(),
        "artifact_sha256": artifact_hashes,
        "warmup": warmup,
        "websocket_connections_at_finish": feed.status_snapshot(),
        "first_market_start_utc": first_start.isoformat(),
        "finished_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "market_count_requested": SMOKE_MARKET_COUNT,
        "market_count_captured": len(captures),
        "consecutive_market_starts": [item["market_start_utc"] for item in captures],
        "captures": captures,
        "official_outcomes": settlements,
        "missing_or_late_outcomes": [item["market_slug"] for item in captures if item["market_slug"] not in settlements],
        "errors": errors,
        "orders_enabled": False,
        "private_credentials_used": False,
        "included_in_prospective_quality_sample": False,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"public_smoke_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=prospective.shadow._json_default), encoding="utf-8")
    feed.stop()
    session.close()
    runtime_connection.close()
    print(f"[public-smoke] report={output.relative_to(ROOT).as_posix()}", flush=True)


if __name__ == "__main__":
    run_smoke()
