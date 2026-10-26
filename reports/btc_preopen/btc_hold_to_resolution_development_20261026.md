# BTC buy-and-hold-to-resolution development economics — 2026-10-26

## Outcome

The latest user instruction makes **buy UP/DOWN, then hold through official Gamma resolution and payout** the active BTC strategy (D0018). This report reuses the existing frozen E0031 confirmation ledger and computes the required $100 accounts for fixed $5 gross and 5% of free cash. It does not run a new model scorer or use holdout outcomes.

The available economic sample is only **1,143 confirmation markets**, from 2026-04-15 17:35 through 2026-05-18 09:00 UTC. It is 2.22% of the 51,430 confirmation markets with official Gamma outcomes. It does not cover full development, so the result remains partial and previously exposed.

## Frozen identities and inputs

Source run identity is `ce108d91f777f7cb861d344f29bc0442f0fa818008cf99c3e62194874456530b`; source E0031 v3 protocol SHA-256 is `16e66363f35390d520e5a3ca7ce58fa07689ac15e66244a313c50f1a29017503`. The full 24 member-model and seven transform hashes remain in that manifest. Candidate sets are identified by their frozen procedure-manifest hashes:

| Candidate | Frozen procedure | Procedure-manifest SHA-256 |
|---|---|---|
| A | BASE70 | `28368afa62f824b5d2592eb61f87cb7a48ba8b66d0afc0a97ccdf6f334c9f1ef` |
| B | BASE70 + simple memory PCA8 | `66eddd315dc38326f6f64959cd4a4fd8d053a6cd189ef7cceac5b3c35bd6a567` |
| C | BASE70 + frozen ESN PCA8 (auxiliary) | `f5ca0f29908e473b7e67be344c9430da0163cfd80f29d8bba9ce3931985f4d45` |
| D | BASE70 + memory + ESN PCA8 | `c4910eb9d4fff03680fe36242c6d338abb2dcbc422803e94bd15c700869e6119` |

Frozen run settings and source hashes are in [the E0036 protocol](../../configs/research/btc_resolution_development_20261026.json), SHA-256 `6e5812ca9fee860313c2e64fdd4a39111a03df55a2da37134df86c17fad04ed6`. The 1,143-row source ledger SHA-256 is `c0432980574e91ecd87529c8367e89f1b08c0cc119b81c345bb7de5acc8fc1a4`; T−60 snapshot SHA-256 is `1df26fe04d2e8d0f0fb9a3820184ce4a5b10c42383b8d424975dde44ef6a84b9`; T−59 ask-ladder SHA-256 is `c90bb2364fff225b6e5047ff761f5ea150dda35afdd63a88ed0fef6f6bd6c941`.

The T−60 and T−59 parquet reads applied Arrow filters for the frozen confirmation IDs and development cutoff before materializing projected quote/quality fields. The T−59 source file spans into holdout; no holdout outcomes or target columns were projected or used. The runner hashes the frozen source files and fails closed on mismatches.

## Existing calculation versus new calculation

- **Already calculated by E0031:** frozen A/B/C/D prediction scores on official Gamma labels; the 1,143-market fixed-$5 independent trade PnL and a $100 fixed-$5 resolution portfolio under the same release assumption.
- **Calculated here:** the 5%-of-free-cash account on the same common confirmation sample; event-level fixed-$5 and 5% ledgers; cash-lock reconciliation, fee totals, peak locked cost, cost-basis equity drawdown, and unresolved-position count. The fixed-$5 ending equities match E0031 exactly within floating-point tolerance.
- **Still missing for complete development economics:** entry source coverage over the full [2025-07-01, 2026-07-01) development interval; the exact market/token/time gap manifest for T−60 decisions; and a cutoff audit for feature, training, transform, selection and calibration provenance. Saved per-prediction `prediction_available_at_utc` telemetry is absent and was not fabricated. These are T0009, not a missing sale rule.

## Fixed-$5 results

Independent trades are a diagnostic that ignores account funding limits. The funded account starts at $100, uses no credit/deposits, and may reject a valid signal when the cash debit exceeds free cash. Fees are the equivalent total for accepted funded trades; V1 share fees reduce shares rather than cash. Drawdown is peak-to-trough in free cash plus locked gross cost basis and excludes intratrade mark-to-market.

| Candidate | Independent trades | Independent PnL | Funded trades | Cash rejections | Accepted fees (equivalent) | Payouts | Ending equity / PnL | Max drawdown | Peak locked gross |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 424 | −$108.80 | 285 | 139 | $50.49 | $1,353.44 | $4.58 / −$95.42 | 96.21% | $10.00 |
| B | 396 | +$28.01 | 396 | 0 | $69.90 | $2,052.53 | $128.01 / +$28.01 | 55.99% | $10.00 |
| C | 375 | +$20.42 | 375 | 0 | $66.19 | $1,939.23 | $120.42 / +$20.42 | 69.32% | $10.00 |
| D | 380 | +$22.66 | 380 | 0 | $67.11 | $1,965.82 | $122.66 / +$22.66 | 72.53% | $10.00 |

All accepted positions received a single payout and settled before the development cutoff under the assumed payout-availability rule. No locked cost remained at that cutoff. B/C/D therefore end positive on this sparse subset, but this is not complete-development validation or independent confirmation. B versus D changed participation by −$5.35 in the prior E0031 attribution; the candidate comparison remains unresolved.

## 5%-of-free-cash control

Each candidate has its own $100 account. At each T−60 decision, requested gross notional is 5% of available free cash after only already-available payouts. The frozen positive-expected-net-payout rule is evaluated at that notional using the T−60 native best ask and displayed size; T−59 full ask depth and the existing fee helper determine execution. The historical Gamma minimum is unavailable, so the accepted $5 fallback remains in force.

| Candidate | Accepted trades | Below-$5 fallback rejections | Other no-positive-EV decisions | Cash rejections | Fee equivalent | Payouts | Ending equity / PnL | Max drawdown | Peak locked gross |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1 | 1,137 | 5 | 0 | $0.18 | $0.00 | $95.00 / −$5.00 | 5.00% | $5.00 |
| B | 1 | 1,137 | 5 | 0 | $0.18 | $0.00 | $95.00 / −$5.00 | 5.00% | $5.00 |
| C | 2 | 1,121 | 20 | 0 | $0.36 | $9.46 | $99.24 / −$0.76 | 5.00% | $5.22 |
| D | 1 | 1,136 | 6 | 0 | $0.18 | $0.00 | $95.00 / −$5.00 | 5.00% | $5.00 |

There were no T−60 size failures, T−59 no-fills or overdrafts. After the first losing payout, free cash fell to $95, so the next requested 5% stake was below the $5 minimum fallback; that account could no longer trade. This is a mechanical consequence of applying both D0006 sizing and D0004 minimum-order rules, not evidence that a practical venue minimum is $5 for those dates.

## Settlement and causal accounting

For every accepted position, the ledger records market end (`market_start + 5 minutes`), Gamma `resolved_at_utc` (official outcome-known time), and assumed payout availability separately. The reused frozen E0031 rule is `max(resolved_at_utc, market end) + 60 seconds`. Cost stays locked until that assumed availability time; the winning token pays its net shares once and the losing token pays zero. No sale or sale fee is recorded. The new ledger tests exercise both winning tokens, losing tokens, fee debit, locked balance, one-time payout and release timing.

Actual redemption-availability timestamps are not archived. The 60-second rule is an idealized assumption and is not verified to conservatively bound real redemption latency. The saved predictions also have no historical per-row T−60 readiness timestamps; the diagnostic assumes they were available at the nominal cutoff and is not live-readiness evidence. D0005's full-pipeline independent validation limitation and D0012/E0033's historical rule-source limits remain.

E0030/E0031's adjusted predictive intervals for the primary B−A and D−B comparisons include zero. No economic PnL confidence interval is reported: trades are not independent, and the saved protocol does not define a bootstrap for the stateful account path. The known negative results for earlier model families and other variants remain in force.

## Holdout plan (not run)

This sparse sample does not close the development gate, so it does not yet justify opening the protected holdout for another economic run. If T0009 completes and the development result warrants it, the separate plan is:

1. Keep the frozen holdout at [2026-07-01, 2026-10-07) UTC under D0001. Report its prior exposure; do not describe it as unseen.
2. Before reading any holdout outcome rows, freeze the selected candidate using development only, its artifact/procedure hashes, the entry policy, both D0006 sizing accounts, fee rules, the common executable sample, settlement timing, metrics and uncertainty method. Do not tune model, entry, threshold or calibration on holdout.
3. Verify exact T−60 decision books and T−59 native ask depth from existing PMXT/Kacho and the existing market/token mapping. E0031 currently has **zero exact T−60 holdout economic rows**; if a causally valid source cannot be established, do not run or substitute another period. Do not buy/acquire an archive in this task.
4. Use official Gamma outcomes for the one-time holdout accounting. Keep the result separate from development; a positive result still does not activate live/shadow, which requires separate explicit user authorization under D0017.

No new holdout data or outcomes, model scores, tuning, sale-policy research, live/shadow process, DNS check, collector, or order was run here.

## Reproduction and validation

- Frozen protocol: [btc_resolution_development_20261026.json](../../configs/research/btc_resolution_development_20261026.json), SHA-256 `6e5812ca9fee860313c2e64fdd4a39111a03df55a2da37134df86c17fad04ed6`.
- Runner SHA-256: `11fd99afa71a787cae135e1334416d9ccbc4c7bb4e0275edd1de17a361b28414`. Cached summary SHA-256: `56a1d8e137657a7af197e9286ed7f996debba8f99e7c98d411f136573000684e`; local event ledgers and run manifest are under `data/analysis/polymarket/BTC/btc_resolution_development_20261026_6e5812ca9fee/` (ignored analysis outputs).
- Run: `python run_btc_resolution_development_20261026.py` — 1,143 confirmation markets, four candidates, two account controls; 0.45 s measured runner time after Python startup.
- Settlement regression: `python -m unittest discover -s tests -p "test_btc_resolution_portfolio.py"` — 3/3 PASS.
- Existing replay/accounting regressions: `python -m unittest discover -s tests -p "test_btc_preopen_economic_replay.py"` — 24/24 PASS.
- Full regression suite: `python -m unittest discover -s tests` — 683/683 PASS in 51.332 seconds (exit 0).

**Next action: T0009**, verify full-development entry coverage and causal provenance using the existing local sources, then publish the exact gap manifest before deciding whether any further calculation is possible.
