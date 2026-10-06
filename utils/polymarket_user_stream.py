"""Authenticated Polymarket order and trade event capture."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from websocket import WebSocketApp


USER_STREAM_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"


def _value(mapping, *keys, default=None):
    for key in keys:
        if isinstance(mapping, dict) and key in mapping and mapping[key] is not None:
            return mapping[key]
        if mapping is not None and hasattr(mapping, key):
            value = getattr(mapping, key)
            if value is not None:
                return value
    return default


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _source_time(value, *, numeric_unit=None):
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) or str(value).strip().isdigit():
        number = float(value)
        unit = numeric_unit or ("ms" if abs(number) >= 1e12 else "s")
        scale = 1000.0 if unit == "ms" else 1.0
        return datetime.fromtimestamp(number / scale, timezone.utc).isoformat()
    try:
        from pandas import Timestamp

        timestamp = Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _source_epoch(value):
    normalized = _source_time(value)
    if normalized is None:
        return None
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def _plain(value):
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value


class PolymarketUserTradeMonitor:
    """Capture account order/trade updates without blocking prediction callbacks."""

    def __init__(
            self,
            client,
            journal_path,
            *,
            websocket_factory=WebSocketApp,
            stream_url=USER_STREAM_URL,
            reconcile_window_seconds=24 * 60 * 60,
            heartbeat_seconds=10.0,
            reconnect_initial_seconds=1.0,
            reconnect_max_seconds=30.0,
            clock_utc=_utc_now,
            monotonic_ns=time.monotonic_ns,
    ):
        self.client = client
        self.journal_path = Path(journal_path)
        self.websocket_factory = websocket_factory
        self.stream_url = str(stream_url)
        self.reconcile_window_seconds = int(reconcile_window_seconds)
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.reconnect_initial_seconds = float(reconnect_initial_seconds)
        self.reconnect_max_seconds = float(reconnect_max_seconds)
        self.clock_utc = clock_utc
        self.monotonic_ns = monotonic_ns
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._thread = None
        self._background_threads = []
        self._ws = None
        self._order_links = {}
        self._orders = {}
        self._trades = {}
        self._unmatched_trade_events = []
        self._load_journal_state()

    def _load_journal_state(self):
        if not self.journal_path.is_file():
            return
        try:
            with self.journal_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("record_type") == "submission_link":
                        order_id = str(record.get("order_id") or "")
                        if order_id:
                            self._order_links[order_id] = record
                    elif record.get("record_type") == "trade_leg":
                        leg_id = str(record.get("trade_leg_id") or "")
                        if leg_id:
                            self._trades[leg_id] = record
                    elif record.get("record_type") == "order_state":
                        order_id = str(record.get("order_id") or "")
                        if order_id:
                            self._orders[order_id] = record
        except OSError:
            return

    def _append(self, record):
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(
                record,
                ensure_ascii=True,
                separators=(",", ":"),
                default=str,
            ))
            stream.write("\n")

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="polymarket-user-trade-stream",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout=5.0):
        self._stop_event.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=max(float(timeout), 0.0))
        with self._lock:
            background_threads = list(self._background_threads)
        deadline = time.monotonic() + max(float(timeout), 0.0)
        for thread in background_threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def _start_background(self, target, *, name):
        with self._lock:
            self._background_threads = [
                thread for thread in self._background_threads if thread.is_alive()
            ]
            thread = threading.Thread(target=target, name=name, daemon=True)
            self._background_threads.append(thread)
            thread.start()

    def register_order(
            self,
            order_id,
            *,
            decision_id,
            attempt_id,
            condition_id=None,
            asset_id=None,
    ):
        order_id = str(order_id or "").strip()
        if not order_id:
            return
        link = {
            "record_type": "submission_link",
            "order_id": order_id,
            "decision_id": str(decision_id or ""),
            "attempt_id": str(attempt_id or ""),
            "condition_id": str(condition_id or ""),
            "asset_id": str(asset_id or ""),
            "linked_at_utc": self.clock_utc(),
        }
        with self._lock:
            self._order_links[order_id] = link
            pending = list(self._unmatched_trade_events)
            self._unmatched_trade_events.clear()
            previous_legs = [
                dict(row)
                for row in self._trades.values()
                if row.get("order_id") == order_id
            ]
        self._append(link)
        for row in previous_legs:
            row["decision_id"] = link["decision_id"]
            row["attempt_id"] = link["attempt_id"]
            with self._lock:
                self._trades[row["trade_leg_id"]] = row
            self._append(row)
        for event, receive_time in pending:
            self._handle_trade(event, receive_time, source="stream_buffered")

    def _auth_payload(self):
        credentials = getattr(self.client, "creds", None)
        if credentials is None:
            raise RuntimeError("Authenticated user stream requires client.creds")
        auth = {
            "apiKey": _value(credentials, "api_key", "apiKey", "key"),
            "secret": _value(credentials, "api_secret", "secret"),
            "passphrase": _value(credentials, "api_passphrase", "passphrase"),
        }
        if not all(auth.values()):
            raise RuntimeError("CLOB API credentials are incomplete for user stream")
        return {
            "auth": auth,
            "type": "user",
        }

    def _on_open(self, ws):
        with self._lock:
            self._ws = ws
        ws.send(json.dumps(self._auth_payload(), separators=(",", ":")))
        if self._stop_event.is_set():
            return
        self._start_background(
            lambda: self._heartbeat(ws),
            name="polymarket-user-stream-heartbeat",
        )
        self._start_background(
            self._reconcile_after_connect,
            name="polymarket-user-stream-reconcile",
        )

    def _heartbeat(self, ws):
        while not self._stop_event.wait(self.heartbeat_seconds):
            with self._lock:
                if self._ws is not ws:
                    return
            try:
                ws.send("PING")
            except Exception:
                return

    def _on_message(self, _ws, message):
        if message == "PONG":
            return
        try:
            payload = json.loads(message) if isinstance(message, str) else message
        except (TypeError, json.JSONDecodeError):
            return
        rows = payload if isinstance(payload, list) else [payload]
        receive_time = self.clock_utc()
        received_monotonic_ns = self.monotonic_ns()
        for event in rows:
            if not isinstance(event, dict):
                continue
            self._handle_event(
                event,
                receive_time,
                received_monotonic_ns,
                source="stream",
            )

    def _on_error(self, _ws, error):
        self._append({
            "record_type": "stream_error",
            "received_at_utc": self.clock_utc(),
            "error": str(error),
        })

    def _on_close(self, ws, code, reason):
        with self._lock:
            if self._ws is ws:
                self._ws = None
        self._append({
            "record_type": "stream_disconnect",
            "received_at_utc": self.clock_utc(),
            "close_code": code,
            "close_reason": str(reason or ""),
        })

    def _run(self):
        delay = self.reconnect_initial_seconds
        while not self._stop_event.is_set():
            try:
                ws = self.websocket_factory(
                    self.stream_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                ws.run_forever()
            except Exception as exc:
                self._on_error(None, exc)
            if self._stop_event.is_set():
                break
            self._stop_event.wait(delay)
            delay = min(delay * 2.0, self.reconnect_max_seconds)

    def _trade_params(self):
        from py_clob_client_v2.clob_types import TradeParams

        after = int(time.time()) - self.reconcile_window_seconds
        return TradeParams(after=after)

    def _reconcile_after_connect(self):
        try:
            open_orders = self.client.get_open_orders()
            recent_trades = self.client.get_trades(params=self._trade_params())
        except Exception as exc:
            self._append({
                "record_type": "reconciliation_error",
                "received_at_utc": self.clock_utc(),
                "error": str(exc),
            })
            return
        receive_time = self.clock_utc()
        receive_monotonic_ns = self.monotonic_ns()
        for order in open_orders or []:
            self._handle_event(
                {"event_type": "order", **(_plain(order) or {})},
                receive_time,
                receive_monotonic_ns,
                source="rest_open_orders",
            )
        for trade in recent_trades or []:
            self._handle_event(
                {"event_type": "trade", **(_plain(trade) or {})},
                receive_time,
                receive_monotonic_ns,
                source="rest_recent_trades",
            )
        self._append({
            "record_type": "reconciliation_complete",
            "received_at_utc": receive_time,
            "open_order_count": len(open_orders or []),
            "recent_trade_count": len(recent_trades or []),
        })

    def _handle_event(self, event, receive_time, receive_monotonic_ns, *, source):
        event_type = str(
            _value(event, "event_type", "eventType", "type", default="")
        ).lower()
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else event
        if event_type == "order":
            self._handle_order(payload, receive_time, receive_monotonic_ns, source)
        elif event_type == "trade" or str(event.get("type", "")).upper() == "TRADE":
            self._handle_trade(payload, receive_time, source=source,
                               receive_monotonic_ns=receive_monotonic_ns)

    def _handle_order(self, event, receive_time, receive_monotonic_ns, source):
        order_id = str(_value(event, "id", "order_id", "orderId", default="") or "")
        if not order_id:
            return
        with self._lock:
            link = self._order_links.get(order_id, {})
            previous = self._orders.get(order_id, {})
            source_raw = _value(event, "timestamp")
            source_at = _source_time(source_raw)
            previous_at = previous.get("exchange_timestamp_utc")
            source_epoch = _source_epoch(source_raw)
            previous_epoch = _source_epoch(previous_at)
            stale_rest_snapshot = (
                source.startswith("rest")
                and str(previous.get("source", "")).startswith("stream")
                and source_epoch is not None
                and previous_epoch is not None
                and source_epoch <= previous_epoch
            )
            if (
                stale_rest_snapshot
                or (
                    source_epoch is not None
                    and previous_epoch is not None
                    and source_epoch < previous_epoch
                )
            ):
                self._append({
                    "record_type": "order_observation",
                    "order_id": order_id,
                    "received_at_utc": receive_time,
                    "source": source,
                    "stale": True,
                    "source_event": event,
                })
                return
            row = {
                "record_type": "order_state",
                "order_id": order_id,
                "decision_id": link.get("decision_id", ""),
                "attempt_id": link.get("attempt_id", ""),
                "condition_id": _value(event, "market", "conditionId", "condition_id", default=link.get("condition_id", "")),
                "asset_id": _value(event, "asset_id", "assetId", default=link.get("asset_id", "")),
                "order_event_type": _value(event, "type", "orderEventType", "order_event_type"),
                "status": _value(event, "status"),
                "side": _value(event, "side"),
                "price": _value(event, "price"),
                "original_size": _value(event, "originalSize", "original_size", "size"),
                "size_matched": _value(event, "sizeMatched", "size_matched"),
                "exchange_timestamp_raw": source_raw,
                "exchange_timestamp_utc": source_at,
                "received_at_utc": receive_time,
                "received_monotonic_ns": int(receive_monotonic_ns),
                "source": source,
            }
            self._orders[order_id] = row
        self._append(row)

    def _own_trade_legs(self, event):
        credentials = getattr(self.client, "creds", None)
        own_key = str(_value(credentials or {}, "api_key", "apiKey", "key", default=""))
        taker_id = str(_value(event, "taker_order_id", "takerOrderId", default="") or "")
        trader_side = str(_value(event, "trader_side", "traderSide", default="") or "").upper()
        with self._lock:
            links = dict(self._order_links)
        legs = []
        if trader_side == "TAKER" or taker_id in links:
            if taker_id:
                legs.append((taker_id, {
                    "amount": _value(event, "size"),
                    "price": _value(event, "price"),
                    "fee_rate_bps": _value(event, "fee_rate_bps", "feeRateBps"),
                    "fee_amount": _value(event, "fee", "fee_amount", "feeAmount"),
                    "side": _value(event, "side"),
                    "asset_id": _value(event, "asset_id", "assetId"),
                }))
        makers = _value(event, "maker_orders", "makerOrders", default=[]) or []
        for maker in makers:
            maker = _plain(maker) or {}
            order_id = str(_value(maker, "order_id", "orderId", default="") or "")
            owner = str(_value(maker, "owner", default="") or "")
            if order_id and (order_id in links or (own_key and owner == own_key)):
                legs.append((order_id, {
                    "amount": _value(maker, "matched_amount", "matchedAmount"),
                    "price": _value(maker, "price"),
                    "fee_rate_bps": _value(maker, "fee_rate_bps", "feeRateBps"),
                    "fee_amount": _value(maker, "fee", "fee_amount", "feeAmount"),
                    "side": _value(maker, "side"),
                    "asset_id": _value(maker, "asset_id", "assetId"),
                }))
        unique = {}
        for order_id, values in legs:
            if order_id:
                unique[order_id] = values
        return list(unique.items())

    def _handle_trade(
            self,
            event,
            receive_time,
            *,
            source,
            receive_monotonic_ns=None,
    ):
        trade_id = str(_value(event, "id", "trade_id", "tradeId", default="") or "")
        if not trade_id:
            return
        legs = self._own_trade_legs(event)
        if not legs:
            if source in {"stream", "stream_buffered"}:
                with self._lock:
                    self._unmatched_trade_events.append((event, receive_time))
            return
        status = _value(event, "status")
        source_match = _value(event, "matched_at", "match_time", "matchedAt")
        source_update = _value(event, "updated_at", "last_update", "updatedAt", "lastUpdate")
        exchange_timestamp = _value(event, "timestamp")
        source_time = _source_time(source_match) or _source_time(exchange_timestamp)
        update_time = _source_time(source_update)
        incoming_update_epoch = _source_epoch(source_update or exchange_timestamp)
        maker_orders = _value(event, "maker_orders", "makerOrders", default=[]) or []
        maker_order_ids = [
            str(_value(_plain(maker) or {}, "order_id", "orderId", default="") or "")
            for maker in maker_orders
        ]
        for order_id, values in legs:
            trade_leg_id = f"{trade_id}:{order_id}"
            with self._lock:
                previous = self._trades.get(trade_leg_id, {})
                previous_update_epoch = _source_epoch(
                    previous.get("exchange_updated_at_utc")
                    or previous.get("exchange_timestamp_utc")
                )
                stale_rest_snapshot = (
                    source.startswith("rest")
                    and str(previous.get("source", "")).startswith("stream")
                    and incoming_update_epoch is not None
                    and previous_update_epoch is not None
                    and incoming_update_epoch <= previous_update_epoch
                )
                if (
                    stale_rest_snapshot
                    or (
                        incoming_update_epoch is not None
                        and previous_update_epoch is not None
                        and incoming_update_epoch < previous_update_epoch
                    )
                ):
                    self._append({
                        "record_type": "trade_observation",
                        "trade_id": trade_id,
                        "order_id": order_id,
                        "received_at_utc": receive_time,
                        "source": source,
                        "stale": True,
                        "source_event": event,
                    })
                    continue
                link = self._order_links.get(order_id, {})
                status_history = list(previous.get("status_history", []))
                if status is not None and (not status_history or status_history[-1] != str(status)):
                    status_history.append(str(status))
                row = {
                    "record_type": "trade_leg",
                    "trade_leg_id": trade_leg_id,
                    "trade_id": trade_id,
                    "order_id": order_id,
                    "decision_id": link.get("decision_id", previous.get("decision_id", "")),
                    "attempt_id": link.get("attempt_id", previous.get("attempt_id", "")),
                    "condition_id": _value(event, "market", "condition_id", "conditionId", default=previous.get("condition_id", "")),
                    "asset_id": values.get("asset_id") or _value(event, "asset_id", "assetId", default=previous.get("asset_id", "")),
                    "side": values.get("side") or _value(event, "side", default=previous.get("side", "")),
                    "price": values.get("price") if values.get("price") is not None else _value(event, "price", default=previous.get("price")),
                    "quantity": values.get("amount") if values.get("amount") is not None else _value(event, "size", default=previous.get("quantity")),
                    "fee_rate_bps": values.get("fee_rate_bps") if values.get("fee_rate_bps") is not None else _value(event, "fee_rate_bps", "feeRateBps", default=previous.get("fee_rate_bps")),
                    "fee_amount": values.get("fee_amount") if values.get("fee_amount") is not None else _value(event, "fee", "fee_amount", "feeAmount", default=previous.get("fee_amount")),
                    "status": status if status is not None else previous.get("status"),
                    "status_history": status_history,
                    "is_terminal": str(status or previous.get("status", "")).upper() in {"TRADE_STATUS_CONFIRMED", "TRADE_STATUS_FAILED", "CONFIRMED", "FAILED"},
                    "exchange_timestamp_raw": exchange_timestamp,
                    "exchange_timestamp_utc": _source_time(exchange_timestamp),
                    "exchange_match_time_raw": source_match,
                    "exchange_match_at_utc": source_time,
                    "exchange_update_time_raw": source_update,
                    "exchange_updated_at_utc": update_time,
                    "transaction_hash": _value(event, "transaction_hash", "transactionHash", default=previous.get("transaction_hash")),
                    "maker_order_ids": maker_order_ids,
                    "received_at_utc": receive_time,
                    "received_monotonic_ns": int(receive_monotonic_ns or self.monotonic_ns()),
                    "source": source,
                    "source_event": event,
                }
                self._trades[trade_leg_id] = row
            self._append(row)

    def handle_message_for_test(self, event, *, receive_time=None, source="stream"):
        self._handle_event(
            event,
            receive_time or self.clock_utc(),
            self.monotonic_ns(),
            source=source,
        )

    def snapshot(self):
        with self._lock:
            return {
                "orders": dict(self._orders),
                "trades": dict(self._trades),
                "order_links": dict(self._order_links),
            }
