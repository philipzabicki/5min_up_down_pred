# Live latency telemetry

`run.py` keeps the existing per-decision CSV records and console log. It does not wait for an order fill or add a separate logging service.

## Where to read it

- Each run writes a trade record under `data/live/<ASSET>/trade/` with a name beginning `live_trade_polymarket_`. The new timing and provenance fields are columns in that CSV.
- Shared decision and market fields are also written to `data/live/<ASSET>/polymarket_5m.csv`.
- The console log path is printed on startup. Search it for `[latency_summary]`; each completed prediction prints a compact JSON summary for the current run.

Use `decision_id` to join fields for one decision. `pm_condition_id`, `pm_up_token_id`, and `pm_down_token_id` identify the market and outcome tokens. `pm_model_hash` and `pm_policy_hash` identify the loaded model and policy. `pm_order_id` is populated only when the CLOB response contains an order ID.

## Event timestamps and durations

`pm_market_start_at_utc` is the market start from Gamma when available; `pm_start_time_source` shows whether it came from `gamma_startDate` or the scheduled slug bucket fallback. `pm_nominal_decision_at_utc` is T−60 seconds.

The Binance candle fields record exchange source time and local receive time separately (`ws_price_*`, `ws_volume_*`). `required_inputs_ready_at_utc`, `features_ready_at_utc`, `prediction_ready_at_utc`, `policy_decision_ready_at_utc`, and `cycle_completed_at_utc` mark local milestones. Cycle completion is recorded after market lookup, the policy decision, and the submit call return; CSV persistence and Telegram notification are outside that interval. Existing `feature_prep_ms`, `feature_vector_ms`, `model_predict_ms`, `policy_compute_ms`, `market_lookup_ms`, `submit_order_ms`, and `execution_ms` are local durations measured with a monotonic clock.

The current market lookup reads the UP and DOWN books through REST `/book`. For each side, `pm_*_book_request_started_at_utc` and `pm_*_book_received_at_utc` bracket the request; `pm_*_book_source_at_utc` and `pm_*_book_source_timestamp_raw` preserve the API timestamp if supplied. `pm_book_data_origin` identifies REST, and `pm_book_stream_sync_status=not_connected_rest_snapshot` means no live book stream is maintained. `pm_up_best_bid`, `pm_up_best_ask`, `pm_down_best_bid`, and `pm_down_best_ask` are the levels used for the policy decision.

`pm_submit_call_started_at_utc` and `pm_submit_call_completed_at_utc` bracket the synchronous client call. `pm_submit_response_received_at_utc` records a response returned to this process. The existing `submit_order_ms` duration uses a monotonic clock. These fields do not reveal when bytes left the host or when the exchange accepted the order.

## Reading `[latency_summary]`

Every stage under `stages_from_nominal_decision` reports `n`, p50, p95, p99, maximum, count over the 1,000 ms budget, and count of negative timestamp differences. Values are wall-clock differences from `pm_nominal_decision_at_utc`; they are not a synchronized cross-service clock measurement. Blank/unavailable stages have `n=0` and null percentiles.

- `cycle_completed` includes all completed prediction cycles.
- `submit_call_started`, `submit_call_completed`, and `client_response_received` include only records with those timestamps. Do not use all-cycle latency as an order-attempt latency.
- `transport_send_observed` is currently unavailable. The CLOB client does not expose a transport-send timestamp to this code.
- `order_ack_source_event`, `fill_event_source`, and `fill_event_received` are currently unavailable because no source-timestamped user order/fill stream is connected.

The separate counts distinguish cycles and their outcomes, calls attempted and their statuses, client responses, unique order IDs, positive fill amounts reported in API responses, and independently timestamped fill-event records. `pm_response_filled_stake_usdc` contains only the amount parsed from the response. The legacy `filled_stake_usdc` record field can fall back to the committed requested amount, so do not use it to count response-reported fills. Even a positive amount in the API response is not an independent fill event or a fill timestamp. `cycle_outcome`, `transaction_skip_reason`, and `pm_order_status` show whether a cycle skipped, a call returned/rejected, or submission failed. A submit error can have a call start/end but no response-received time.

`pm_auth_clock_sync_status`, `pm_auth_clock_sync_at_utc`, and `pm_auth_clock_offset_seconds_estimate` describe the last available sample from the CLOB `/time` endpoint. The offset is a rough endpoint/host comparison: this code does not record a calibrated NTP state, request round-trip uncertainty, or an error bound. Treat missing source, send, ACK, and fill timestamps as unavailable rather than inferring them from another event.

Do not sum p50/p95/p99 values from separate stages. Their samples need not be the same decisions, and adding quantiles does not produce an end-to-end quantile.
