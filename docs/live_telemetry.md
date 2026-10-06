# Live latency telemetry

`run.py` keeps the existing per-decision CSV records and console log. It does not wait for an order fill. In live mode with order submission enabled, a background authenticated user stream records Polymarket order and trade updates separately. The wire subscription follows the current [Polymarket Real-Time Order Updates protocol](https://docs.polymarket.com/trading/realtime-order-updates): authenticate on `/ws/user`, subscribe with `type=user`, and send `PING` every 10 seconds.

## Where to read it

- Each run writes a trade record under `data/live/<ASSET>/trade/` with a name beginning `live_trade_polymarket_`. The new timing and provenance fields are columns in that CSV.
- Live order and trade events are appended to `data/live/<ASSET>/trade/polymarket_user_events_<ASSET>.jsonl`. A reconnect triggers REST reconciliation of open orders and recent trades; duplicate trade legs are updated by trade/order ID.
- Shared decision and market fields are also written to `data/live/<ASSET>/polymarket_5m.csv`.
- The console log path is printed on startup. Search it for `[latency_summary]`; each completed prediction prints a compact JSON summary for the current run.

Use `decision_id` and `pm_attempt_id` to join fields for a decision and submission attempt. A user event includes `order_id`, `attempt_id`, `decision_id`, and `condition_id` when those values are known. `pm_condition_id`, `pm_up_token_id`, and `pm_down_token_id` identify the market and outcome tokens. `pm_model_hash` and `pm_policy_hash` identify the loaded model and policy. `pm_order_id` records the CLOB response order ID; the event journal can also capture order IDs reported by the user stream.

## Event timestamps and durations

`pm_market_start_at_utc` is the market start from Gamma when available; `pm_start_time_source` shows whether it came from `gamma_startDate` or the scheduled slug bucket fallback. `pm_nominal_decision_at_utc` is T−60 seconds.

The Binance candle fields record exchange source time and local receive time separately (`ws_price_*`, `ws_volume_*`). `required_inputs_ready_at_utc`, `features_ready_at_utc`, `prediction_ready_at_utc`, `policy_decision_ready_at_utc`, and `cycle_completed_at_utc` mark local milestones. Cycle completion is recorded after market lookup, the policy decision, and the submit call return; CSV persistence and Telegram notification are outside that interval. Existing `feature_prep_ms`, `feature_vector_ms`, `model_predict_ms`, `policy_compute_ms`, `market_lookup_ms`, `submit_order_ms`, and `execution_ms` are local durations measured with a monotonic clock.

The current market lookup reads the UP and DOWN books through REST `/book`. For each side, `pm_*_book_request_started_at_utc` and `pm_*_book_received_at_utc` bracket the request; `pm_*_book_source_at_utc` and `pm_*_book_source_timestamp_raw` preserve the API timestamp if supplied. `pm_book_data_origin` identifies REST, and `pm_book_stream_sync_status=not_connected_rest_snapshot` means no live book stream is maintained. `pm_up_best_bid`, `pm_up_best_ask`, `pm_down_best_bid`, and `pm_down_best_ask` are the levels used for the policy decision.

`pm_submit_call_started_at_utc` and `pm_submit_call_completed_at_utc` bracket the synchronous client call. `pm_submit_response_received_at_utc` records a response returned to this process. The existing `submit_order_ms` duration uses a monotonic clock. These fields do not reveal when bytes left the host or when the exchange accepted the order.

The event journal stores order state updates as `record_type=order_state` and matched trade legs as `record_type=trade_leg`. Polymarket reports order placement/update/cancellation events and trade states such as matched, mined, confirmed, retrying, and failed. The journal records normalized exchange fields, source (`stream`, `stream_buffered`, `rest_open_orders`, or `rest_recent_trades`), exchange timestamps when present, local `received_at_utc`, and a monotonic receive value for websocket messages; trade legs also retain the raw source event. Trade legs are keyed by `trade_leg_id` (`trade_id:order_id`) so maker partial fills can be associated with the order that belongs to this account. A reconnect also records reconciliation completion or error. Polymarket's protocol says a placement update means the order was accepted; this is an exchange event timestamp, not the local transport-send time or an HTTP response timestamp.

## Reading `[latency_summary]`

Every stage under `stages_from_nominal_decision` reports `n`, p50, p95, p99, maximum, count over the 1,000 ms budget, and count of negative timestamp differences. Values are wall-clock differences from `pm_nominal_decision_at_utc`; they are not a synchronized cross-service clock measurement. Blank/unavailable stages have `n=0` and null percentiles.

- `cycle_completed` includes all completed prediction cycles.
- `submit_call_started`, `submit_call_completed`, and `client_response_received` include only records with those timestamps. Do not use all-cycle latency as an order-attempt latency.
- `transport_send_observed` is currently unavailable. The CLOB client does not expose a transport-send timestamp to this code.
- `order_ack_source_event` means the authenticated `order` placement update, when present; it is not a transport ACK. `fill_event_source` and `fill_event_received` come from authenticated `trade` events. The protocol's `timestamp` is milliseconds since epoch; the event's match/update time fields are kept separately when supplied. The stream may reconnect or omit a source timestamp; use the event journal's `source`, raw event, and receive time to distinguish stream observations from REST reconciliation. This audit did not run live, so it observed no placement update or fill.

The separate counts distinguish cycles and their outcomes, calls attempted and their statuses, client responses, unique order IDs, positive fill amounts reported in API responses, and independently timestamped fill-event records. `pm_response_filled_stake_usdc` contains only the amount parsed from the response. The legacy `filled_stake_usdc` record field can fall back to the committed requested amount, so do not use it to count response-reported fills. A positive API response amount is not itself a fill event or fill timestamp. `cycle_outcome`, `transaction_skip_reason`, and `pm_order_status` show whether a cycle skipped, a call returned/rejected, or submission failed. A submit error can have a call start/end but no response-received time.

`pm_auth_clock_sync_status`, `pm_auth_clock_sync_at_utc`, and `pm_auth_clock_offset_seconds_estimate` describe the last available sample from the CLOB `/time` endpoint. The offset is a rough endpoint/host comparison: this code does not record a calibrated NTP state, request round-trip uncertainty, or an error bound. Treat missing source, send, ACK, and fill timestamps as unavailable rather than inferring them from another event.

Do not sum p50/p95/p99 values from separate stages. Their samples need not be the same decisions, and adding quantiles does not produce an end-to-end quantile.
