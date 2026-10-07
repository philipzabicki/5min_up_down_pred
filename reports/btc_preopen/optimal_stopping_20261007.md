# Optimal stopping continuation — 2026-10-07

The isolated comparison reuses the exact same accepted legacy $5 buys and stakes. Every purchase is independently funded from $100; summed PnL is not a portfolio return. The full portfolio separately replays the frozen legacy exact-$5 entry selector with limited cash and sale proceeds available for later entries.

Kacho full 300-second paths: 4454/4454; exact post-start 5-second decisions: 4454/4454. Pre-start PMXT fresh two-sided BBO counts: `{'11': 4426, '10': 17, '9': 5, '8': 3, '0': 2, '6': 1}`; fresh bid-size counts: `{'11': 4260, '9': 29, '10': 138, '7': 3, '8': 15, '6': 3, '0': 4, '4': 1, '3': 1}`. Full-position exit quotes: pre-start 23931, post-start 126675. The stored T-59 bid has no saved update timestamp or size and is not forward-filled.

## Same buys and stakes

| Exit rule | Trades | Sold | Total PnL | Mean independent bet log return | Sale proceeds |
|---|---:|---:|---:|---:|---:|
| hold_to_resolution | 2285 | 0 | $461.42 | 0.000767 | $0.00 |
| simple_expected_value | 2285 | 2050 | $6.94 | -0.000130 | $11401.73 |
| ridge | 2285 | 1961 | $146.62 | 0.000271 | $11038.09 |
| tree | 2285 | 2207 | $43.94 | 0.000046 | $11400.31 |

## Fold consistency for identical buys

The figures below are per-fold paired diagnostics; each trade is independently funded from $100, so fold PnL is not a single-portfolio return.

| Exit rule | Fold | Trades | Sold | PnL | Mean PnL per trade |
|---|---:|---:|---:|---:|---:|
| hold_to_resolution | 1 | 750 | 0 | $161.69 | $0.22 |
| hold_to_resolution | 2 | 767 | 0 | $-60.80 | $-0.08 |
| hold_to_resolution | 3 | 768 | 0 | $360.53 | $0.47 |
| simple_expected_value | 1 | 750 | 678 | $25.03 | $0.03 |
| simple_expected_value | 2 | 767 | 691 | $-9.46 | $-0.01 |
| simple_expected_value | 3 | 768 | 681 | $-8.63 | $-0.01 |
| ridge | 1 | 750 | 699 | $3.85 | $0.01 |
| ridge | 2 | 767 | 595 | $47.15 | $0.06 |
| ridge | 3 | 768 | 667 | $95.61 | $0.12 |
| tree | 1 | 750 | 698 | $5.34 | $0.01 |
| tree | 2 | 767 | 744 | $28.41 | $0.04 |
| tree | 3 | 768 | 765 | $10.19 | $0.01 |

## Full portfolio with reinvestment

| Exit rule | Trades | Sold | PnL | Mean daily log growth | Cost-basis max DD | Liquidation max DD | Liquidation coverage | Max exposure | Underwater seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hold_to_resolution | 2285 | 0 | $461.42 | 0.091518 | 75.37% | 77.18% | 86.7% | $15.00 | 456894 |
| simple_expected_value | 2285 | 2050 | $6.94 | 0.009108 | 74.41% | 75.09% | 95.7% | $10.00 | 1065295 |
| ridge | 2285 | 1961 | $146.62 | 0.042837 | 55.74% | 55.82% | 94.9% | $10.00 | 614760 |
| tree | 2285 | 2207 | $43.94 | 0.021830 | 42.56% | 43.93% | 96.7% | $10.00 | 831344 |

Kacho provides one-second sampled top bid/ask and top sizes from market start through second 299. Its row timestamp is a sample-time proxy, not a WebSocket receive timestamp. No quotes are forward-filled. A sale requires a fresh bid and enough best-bid size for the full position. The continuation models are trained per isolated $100 position and then applied to each position in the shared portfolio; they do not optimize cross-position utility. Sale fees extend the existing date-based fee mode using the market entry fee rate held constant for five minutes; fees are an explicit model approximation. Sale proceeds are available immediately. At equal timestamps, settlement release is processed first, then a new entry, then exits, so an exit cannot finance a same-time entry.

For liquidation drawdown, every open position must be valued at an executable current bid or a deterministic known payout. Unvalued timestamps are excluded rather than forward-filled; the table reports coverage. Cost-basis drawdown uses the original ledger convention.
