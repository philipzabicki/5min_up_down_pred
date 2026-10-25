# BTC A/B/C/D shadow: historical L2 sources and pilot — 2026-10-23

## Result

The read-only collector and its frozen protocol are implemented. The live pilot did not start: the Binance time endpoint returned HTTP 200, while DNS resolution failed for both Polymarket hosts required to fetch Gamma market identity and CLOB books. Consequently, there are zero live decision windows, no later Gamma settlements, and no background process. The runner stops in preflight before model loading.

The historical TWAP60 execution ledger remains empty. The local PMXT archive ends before TWAP60 began, so the zero-row result is missing raw book history, not a demonstrated selector, index, quality, or provenance rejection. A public Vike sample proves that full-depth snapshots and deltas can reconstruct a T−59 book, but that sample is one October 22 market, outside the historical TWAP60 evaluation period. No historical TWAP60 economics were recomputed.

## Historical source review

The review date is 2026-10-23. It distinguishes executed trades and price series from L2 snapshots and event streams. A trade price, midpoint, or price-history point does not establish the ask price or depth available for a $5 buy.

| Source | Date range | BTC 5m | L2 / depth | Timestamps | T−59 reconstruction | Provenance | Sample | Cost / access | Limits |
|---|---|---|---|---|---|---|---|---|---|
| Local PMXT v2 archive | Local manifest ends at 2026-08-10 00:00 UTC; target TWAP60 begins 2026-08-14 | Present before archive end | Full book snapshots plus `price_change`, trades, tick and status events | `timestamp` source time and `timestamp_received` archive receive time | None in the TWAP60 range: no raw PMXT objects exist there | Recorded from Polymarket CLOB market WebSocket; hourly Parquet files | Local manifests and archive already inspected; no target-period sample | Public, no key; no incremental retrieval can produce the missing dates | Archive stops before the target. Existing selector and quality gates did not reject available TWAP60 rows because no rows were present. [PMXT v2 overview](https://archive.pmxt.dev/docs/v2-data-overview) |
| Vike full-L2 archive | Live manifest generated 2026-10-23 00:47 UTC; date layout covers all 54 target days [2026-08-14, 2026-10-07). BTC/5m family layout covers Aug 22–Oct 6; Aug 14–21 are absent from that family layout. | Yes | `book_events` contains full snapshots, deltas, trades, tick-size and status events. The `l1_quotes` file is derived top-of-book only. | `ts` source event time; `local_ts` receive time; `seq` orders events | Feasible by replaying a full snapshot and all eligible deltas for each token through the target receive time | Captured CLOB WebSocket event records; manifest is regenerated daily; CC BY 4.0 | Free real complete sample for one recent market, both tokens and lifecycle. Replayed below. | Manifest and sample need no key. Full archive files require a free Vike API key; no key was present and no account was created. Family files for 46 target days total 53.94 GB. The eight missing family dates require date-layout files totaling 47.94 GB. Estimated download for this mixed route is 101.88 GB; no monetary price is published for the free-key archive. | Family-specific coverage has an eight-day hole. The 101.9 GB figure covers only the requested window using the manifest layouts, not all data stored in the archive. [Archive guide](https://vike.io/docs/data/guides/polymarket/archive/), [manifest guide](https://vike.io/docs/data/endpoints/polymarket/archive/manifest-json/) |
| PMData | Provider documentation says BTC Up/Down 5m is available since 2026-02-01 | Yes | Incremental L2 snapshots/updates. The documented L2 export contains YES-token events only; PMData documents the symmetric complement relation for deriving the other outcome book. | Source and PMData receive timestamps at microsecond precision | Technically feasible if the target market files and both sides' complement data are available | Polymarket WebSocket captures plus provider receive timestamps | Public docs show BTC 5m examples; no target-period file was downloaded | Requires an account and free API key. A daily BTC 5m L2 archive is 72 quota units; 54 days would be 3,888 units. Monetary pricing/quota sufficiency was not confirmed. No account was created. | One-sided token representation requires careful complement mapping. Coverage is provider-reported, not checked row-by-row here. [Coverage and schema](https://pmdata.dev/docs/historical-data-details/polymarket), [API access](https://pmdata.dev/docs), [L2 FAQ](https://pmdata.dev/faq) |
| Marketlens archive | Provider claims collection since 2026-03-01 | Yes | Full-depth book snapshots about every 60 seconds plus intervening individual price changes; trades and candles are separate records | Price changes are documented at millisecond resolution | Technically feasible where a complete snapshot and all subsequent changes are available | Provider-recorded order books, trades and outcomes | Public BTC 5m sample is 2026-06-15 (288 markets; 3,889 snapshots; 41.6M price changes; 681K trades); it does not overlap TWAP60 | Public sample needs no account. Full archive access is marked Pro in the provider material; exact price was not published in the reviewed pages and no account was opened. | No target-period public sample was available to validate its feed coverage for these exact markets. [Dataset](https://marketlens.trade/polymarket-dataset), [sample and archive description](https://github.com/marketlenstrade/polymarket-historical-data) |
| Official Polymarket CLOB / price history | Live endpoints; price-history records are not historical full-depth books | Yes | `GET /book` returns a current L2 snapshot; `prices-history` returns historical prices, without executable size ladders | Current book has a server timestamp; price history has point timestamps | No retrospective T−59 depth reconstruction from these endpoints alone | Polymarket market/CLOB APIs | No separate historical depth sample | Public GET endpoints; no paid data access | Trades and price points do not substitute for two-sided L2 depth. [Get order book](https://docs.polymarket.com/api-reference/market-data/get-order-book.md), [Get prices history](https://docs.polymarket.com/api-reference/markets/get-prices-history.md) |

Local evidence: [`t59_reassessment_20261007.md`](t59_reassessment_20261007/t59_reassessment_20261007.md) records that PMXT's final local archive hour is 2026-08-10 00:00 UTC and that no TWAP60 PMXT events are archived. The previously available execution ledger's 1,143 markets are all single-price confirmation markets; it contains zero TWAP60 rows. There was no selector/index/quality exclusion to repair locally.

### Vike public-sample reconstruction

The real sample identifies market slug `btc-updown-5m-1792684800`, condition ID `0x12bb694cf822f06d6d7d6dfa90e1d50a9f29612dbed008ee47366e660c63850d`, and these two token IDs:

- `32644521512651101758310957372182971414017953083635576488050266270970929248223`
- `71203883629274059726943582191587572207982450308238658734132305909542135523623`

The sample schema does not identify which token is UP versus DOWN. The Gamma metadata host could not be resolved in this environment, so the sample results below are deliberately labelled Token 1 and Token 2; no outcome orientation is guessed.

The sample covers 2026-10-22 15:55:13.502–16:05:24.088 UTC source time, and 15:55:13.711–16:05:24.088 UTC receive time. It has 579,932 events and 288,997 unique global sequence values from 0 through 288,996, with no missing sequence value. The global sequence is shared across tokens; per-token sequence skips are expected and are not evidence of dropped messages. For each cutoff, the replay filtered both `ts` and `local_ts` to be no later than the cutoff, selected each token's latest full `book` snapshot, then applied that token's later `price_change` events in sequence order. No event from after T−60 or T−59 was used.

Market T is 16:00 UTC, T−60 is 15:59:00 and T−59 is 15:59:01. The latest full snapshots for both tokens are from source time 15:58:57.251 and receive times 15:58:57.281 / 15:58:57.279. At T−60 they were 2.719 / 2.721 seconds old, with 159 subsequent deltas per token. At T−59 they were 3.719 / 3.721 seconds old, with 461 subsequent deltas per token. The complete book state at each cutoff therefore comes from the snapshot plus recorded deltas, not from treating the older snapshot as unchanged.

| Cutoff | Token | Best bid / ask | Best ask shares | Displayed ask depth, USD | $5 gross ask-walk VWAP |
|---|---|---:|---:|---:|---:|
| T−60 | 1 | 0.50 / 0.51 | 1,408.16 | 46,830.4383 | 0.51 |
| T−60 | 2 | 0.49 / 0.50 | 20.50 | 47,456.7759 | 0.50 |
| T−59 | 1 | 0.50 / 0.51 | 1,328.16 | 41,513.0145 | 0.51 |
| T−59 | 2 | 0.49 / 0.50 | 175.50 | 42,036.3921 | 0.50 |

At T−59 both visible ask ladders cover a $5 gross walk entirely at the best ask. Applying the repository's date-assigned 2026 fee estimate (rate 0.07, exponent 1, cash-collateral collection, V2 rounding) gives Token 1 an estimated $0.17150 fee and $5.17150 cash debit; Token 2 gives $0.17500 and $5.17500. These are displayed-book and fee-rule estimates, not Gamma-confirmed per-market fees or actual fills. A displayed book never guarantees execution.

## Shadow collector and runtime

The collector uses the existing `run.py` feature runtime, `PseudoLiveAuditPredictor`, frozen candidate bundle and memory/ESN streaming state. It does not activate the existing trading runner. Its code path uses public Binance WebSocket/REST reads, public Gamma/CLOB GET requests, local SQLite, and frozen LightGBM inference. The script contains no order submission/cancellation/execution API calls or authenticated trading client.

The frozen candidates remain A=BASE70, B=BASE70+simple-memory PCA8, C=BASE70+ESN PCA8, D=BASE70+memory+ESN PCA8. Market T is the exact Gamma start; the decision deadline is T−60; input is the one-minute candle opened at T−120; the book execution observation is T−59. Receipt, event, processing, feature, inference and ready timestamps are recorded. A late signal keeps its actual readiness timestamp and is marked ineligible; it is never backdated. REST gap repair restores feature state with prediction generation disabled. Duplicate input retains its source trace, and a conflicting duplicate fails closed.

Once a market attempt is reachable, SQLite records the Gamma slug, condition ID, token mapping, exact start/end and captured settlement metadata; complete T−60 and T−59 books for both tokens; member and ensemble A/B/C/D probabilities; 5-minute candle inputs and source provenance; later official Gamma outcome and receive time; and $5 gross hypothetical ask-walk depth, fee estimate and cash debit. Actual fill remains `unknown`.

Latency is instrumented for cold model/runtime initialization, frozen dynamic feature replay, checkpoint restoration, warm-up/catch-up fetch and state replay, normal candle storage, BASE70 incremental update, memory/ESN update, BASE70 snapshot, taker-flow features, ensemble inference, and input-receipt-to-ready time. The blocked preflight prevented those model-path latencies from being measured in this session. No synthetic timing is presented as a live pilot result.

Storage policy is frozen in [`btc_candidate_shadow_20261023.json`](../../configs/research/btc_candidate_shadow_20261023.json): commit each received/repaired candle and completed market; checkpoint after each market and hourly; truncate the SQLite WAL at checkpoint; keep the research database without automatic rotation or deletion; hard stop at 50 GB. The planning envelope is 0.01–0.06 GB/day for 1,152 full-depth token snapshots/day plus candle provenance and SQLite overhead. This is an estimate only; live measured growth is unavailable until endpoints resolve.

## Pilot attempt

Ran `python run_btc_candidate_shadow_20261023.py` twice, at 2026-10-23 18:46:51 UTC and 18:54:05 UTC, including after the final instrumentation changes. The final preflight requested:

- `https://fapi.binance.com/fapi/v1/time` — HTTP 200.
- `https://gamma-api.polymarket.com/markets/slug/btc-updown-5m-1792781700` — `NameResolutionError`, Windows `getaddrinfo` error 11002, failed to resolve `gamma-api.polymarket.com`.
- `https://clob.polymarket.com/book?token_id=0` — `NameResolutionError`, Windows `getaddrinfo` error 11002, failed to resolve `clob.polymarket.com`.

Both exact probe responses are preserved as `network_preflight` events in `data/analysis/polymarket/BTC/btc_candidate_shadow_20261023/shadow.sqlite3`. The database has zero predictions, markets, snapshots, attempts, and official outcomes. No Gamma identity, live book, prediction, outcome, or complete window was obtained. The pilot count is 0/3; no collector process remains and there is no PID. A three-window pilot plus later Gamma settlement verification must be run after DNS access to both Polymarket hosts works.

## Running and stopping

From the repository root, start or resume with the same command:

```powershell
python run_btc_candidate_shadow_20261023.py
```

It runs in the foreground. Stop it with `Ctrl+C`; it writes a checkpoint and closes the feed. If preflight fails, it records the errors and exits before model initialization. No process is left running by this session.

Database: `data/analysis/polymarket/BTC/btc_candidate_shadow_20261023/shadow.sqlite3`. Checkpoint flush: hourly and after each completed market. No automatic data deletion or database rotation. The runner's checkpoint identity pins the candidate model/transform bundle, feature order, protocol and runtime source hashes; it refuses resume if these differ.

## Precision plan; no fixed 180-day requirement

The earlier 180-day next step was an arbitrary duration and is superseded here. The existing 1,143-market execution sample is single-price confirmation only, so its estimates are planning sensitivities, not representative TWAP60 economics.

The registered outcome unit is net PnL per $5 gross decision, with no-trade contributing zero. Primary comparisons are frozen A versus cash and paired D versus B. The minimum detectable effects of 0.5%, 1% and 2% of $5 are $0.025, $0.05 and $0.10 per transaction; they are precision scenarios, not trading thresholds. Two-sided family-wise alpha 0.05 is Bonferroni-adjusted across the two primary comparisons (per-test alpha 0.025), with 80% power. Thus `n = ((z0.9875 + z0.80) × sigma / MDE)^2`, where the summed z value is 3.083.

| Comparison | Planning sigma scenarios, USD | n at $0.025 MDE | n at $0.05 MDE | n at $0.10 MDE |
|---|---:|---:|---:|---:|
| A vs cash | 1.00 / 2.50 / 4.887 | 15,209 / 95,051 / 363,211 | 3,803 / 23,763 / 90,803 | 951 / 5,941 / 22,701 |
| D vs B | 1.00 / 1.335 / 2.176 | 15,209 / 27,105 / 72,010 | 3,803 / 6,777 / 18,003 | 951 / 1,695 / 4,501 |

The upper A sigma ($4.887) is the old sample's per-trade realized-PnL SD. For D−B, $2.176 is the B-or-D trade-union SD; $1.335 is the SD across all decision windows. The lower values are explicit sensitivity cases. In the old sample A traded 424 times (12.47/calendar day across 34 days, 33 active days); B 396 (11.65/day); D 380 (11.18/day); and the B-or-D union was 431 (12.68/day). A daily aggregate PnL SD was $10.90, daily lag-1 correlation −0.0455, and largest absolute daily contribution 9.77%. D−B across all 1,143 windows had mean −$0.00468 and SD $1.335; its 431-window trade union SD was $2.176, daily aggregate SD $8.144 and lag-1 correlation +0.0564. These concentration/dependence estimates come from the non-TWAP60 sample only.

For a single final look, the protocol selects the 2% MDE and the conservative old A trade sigma. Under independence, the thresholds are 22,701 A transactions and 4,501 D/B-union transactions. A three-day dependence sensitivity uses a planning rate of six transactions/day and intrablock `rho=0.05`: `m=18` transactions/block, `DE=1+(m−1)rho=1.85`, giving 41,997 and 8,327 transactions. The A transaction count alone corresponds to 7,000 complete UTC days at six trades/day; the separate calendar threshold of 2,334 non-overlapping three-day blocks requires 7,002 complete days and is therefore slightly higher. This dependence assumption is a pre-registered sensitivity, not measured TWAP60 dependence. At 12.5 or 20 trades/day, the same calculation gives different durations; the unadjusted independent A threshold alone implies 1,817 or 1,136 days. No interim PnL looks are allowed. Technical quality may be monitored daily. Run one final analysis only after both transaction thresholds and the selected calendar-block threshold are met; do not stop on interim signs. Do not fit or change models, features, thresholds, ensemble or policy.

## Remaining blockers

1. DNS/network access to `gamma-api.polymarket.com` and `clob.polymarket.com` is required for the three-window live pilot, full latency measurements, and subsequent official Gamma settlement checks.
2. Historical TWAP60 L2 is not present locally. Vike can provide the target window but its BTC/5m family files have an eight-day hole and the documented mixed-layout retrieval is about 101.9 GB behind a free API key. PMData is another possible route but requires an account/key and 3,888 published quota units for 54 daily BTC 5m L2 archives. No external account was created and no history was purchased.
3. The free Vike sample does not identify token 1/2 as UP/DOWN without Gamma metadata; its book execution arithmetic is valid per token, but outcome orientation and official market resolution were not inferred.

## Sources

- [Vike Polymarket bulk archive guide](https://vike.io/docs/data/guides/polymarket/archive/) and [live manifest](https://data.vike.io/archive/manifest.json), checked 2026-10-23.
- [PMXT v2 schema, source and access](https://archive.pmxt.dev/docs/v2-data-overview), checked 2026-10-23; local end date from the previously inspected local archive manifest.
- [PMData Polymarket history](https://pmdata.dev/docs/historical-data-details/polymarket), [API access and quotas](https://pmdata.dev/docs/api-reference), and [YES-only L2 FAQ](https://pmdata.dev/faq), checked 2026-10-23.
- [Marketlens dataset](https://marketlens.trade/polymarket-dataset) and [public sample archive](https://github.com/marketlenstrade/polymarket-historical-data), checked 2026-10-23.
- Polymarket [CLOB current order book](https://docs.polymarket.com/api-reference/market-data/get-order-book.md) and [historical price endpoint](https://docs.polymarket.com/api-reference/markets/get-prices-history.md), checked 2026-10-23.

## 2026-10-25 continuation: timing correction and bounded preflight

The binding timing interpretation remains D0003: source information cutoff at T-60, prediction readiness through T-59, and the T-59 book only tests execution of the already-fixed decision. The shadow collector now records Binance kline close event times separately from Binance event and local receive times; UTC timestamps remain durable, while elapsed durations and paired receive skew use `time.monotonic`. A prediction is eligible only when both input kline close events are no later than T-60 and actual prediction readiness is no later than T-59. A REST book with missing receipt time or source/receipt time after its T-60/T-59 boundary remains recorded but cannot be used to price that historical snapshot. No timestamps are backdated.

Bounded public GET preflight in the repository interpreter, 2026-10-25 22:44 UTC:

| Endpoint | Result | Schema/identity | Consequence |
|---|---|---|---|
| Binance `/fapi/v1/time` | HTTP 200 in 0.494 s | JSON object with integer `serverTime` | Passed transport/schema check |
| Gamma next BTC 5m slug | DNS `NameResolutionError` for `gamma-api.polymarket.com` | No market/token/condition schema available | Pilot gate failed |
| CLOB `/book?token_id=0` reachability probe | DNS `NameResolutionError` for `clob.polymarket.com` in 0.084 s | No HTTP response; no book schema available | Pilot gate failed |

The Gamma lookup used the next scheduled five-minute market start, `2026-10-25T22:45:00Z`; request began at `22:44:41.910015Z`. Since Gamma did not return the current market's token mapping, no CLOB request using a valid current token could be made. The collector was not started, and no market was substituted or retried. Pilot planned/observed/complete/timing-qualified/resolved windows: **3/0/0/0/0**. Actual fills, readiness feasibility, book latency and storage growth remain UNKNOWN. The existing database was checked read-only: two earlier preflight events; zero candles, markets, snapshots, predictions, attempts, outcomes, or checkpoints. No new DB rows were written.

The full suite after T0001-T0003 and timing changes passed: `python -m unittest discover -s tests -v`, 680 tests in 45.157 s, exit code 0. Focused timing tests passed 14/14 candidate-shadow and 19/19 prospective-feed tests. These are code checks, not a live pilot or performance evidence. T0004 remains BLOCKED pending Gamma/CLOB DNS and a valid schema preflight; T0007's code interpretation is resolved but measured live timing is pending that pilot.
