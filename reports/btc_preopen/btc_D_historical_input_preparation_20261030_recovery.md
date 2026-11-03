# E0048: odzysk artefaktow po przekroczeniu budzetu

**Wynik: produkty zmaterializowane; limit 4 godzin przekroczony. E0048 nie jest oznaczane jako zaliczone w ramach zamrozonego budzetu.**

Runner v4 wystartowal `2026-10-30T01:04:44Z`. Dokladny czas zakonczenia procesu nie zostal zachowany; materializacja produktow nastapila co najmniej 342799.9 s po starcie, przy budzecie 14400 s.
Odzysk sprawdzil kompletny checkpoint i gotowe produkty. Nie odczytywal ponownie wierszy zdarzen PMXT; sprawdzil tylko metadane Parquet potrzebne do weryfikacji tozsamosci checkpointu. Nie uruchomiono modelu ani ewaluatora.

## Zakres i pokrycie

Cechy: 141,116 wierszy, 102 cech; panel powtorzony: PASS_EXACT (48 wierszy).
Sloty rynku: 28,224; T-60 obie poprawne strony: 18,649; T-59: 17,704; source-only kompletne: 17,135.
Replay ksiazek: 2,377/2,377 godzin; 416,545,575 zdarzen odtworzonych; 68 brakujacych godzin zrodlowych. Zachowano znana luke E0043: 68/68 godzin.

| Miara pokrycia | Liczba |
|---|---:|
| calendar_slots | 28,224 |
| market_identity_confirmed | 28,222 |
| market_identity_unknown_or_conflict | 2 |
| feature_rows_present | 28,224 |
| feature_values_complete | 28,223 |
| feature_rows_ready | 28,223 |
| flow_exact_key_join_matched | 28,224 |
| t60_both_native_sides_valid | 18,649 |
| t59_both_native_sides_valid | 17,704 |
| t60_entry_hour_available | 27,406 |
| t59_entry_hour_available | 27,406 |
| t60_freshness_window_available | 27,406 |
| t59_freshness_window_available | 27,406 |
| t60_full_source_window | 27,107 |
| t59_full_source_window | 27,107 |
| t60_fee_evidence_known | 25,151 |
| t59_fee_evidence_known | 25,151 |
| source_quality_clean_both_cutoffs | 17,135 |
| source_only_components_complete | 17,135 |

### Rezymy

| Rezym | Sloty | Tozsamosc | Cechy gotowe | FLOW | T-60 | T-59 | Oplata T-60 | Oplata T-59 | Source-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 10,656 | 10,655 | 10,656 | 10,656 | 7,146 | 7,190 | 8,553 | 8,553 | 6,944 |
| TWAP30 | 2,016 | 2,016 | 2,016 | 2,016 | 774 | 771 | 2,010 | 2,010 | 763 |
| TWAP60 | 15,552 | 15,551 | 15,551 | 15,552 | 10,729 | 9,743 | 14,588 | 14,588 | 9,428 |

## Integralnosc i granice uzycia

Checkpoint ksiazek ma zweryfikowana integralnosc i tozsamosc; hash: `b9f3a43fa66769e7cb3e01cb7b27b94f704d6806a0f484815c563440fc86903c`. Hashy wszystkich siedmiu produktow i checkpointu nie zmieniano w miejscu; manifest odzyskiwania: `data/analysis/polymarket/BTC/btc_D_historical_inputs_20261030_v4/input_preparation_recovery_manifest.json`.
Targety, etykiety, wyniki, rozstrzygniecia i payouty nie zostaly odczytane. Predykcje, EV, fille, transakcje, PnL i metryki jakosci: 0. `evaluation_authorized=false`, `activated=false`, `HOLDOUT_NOT_RUN`; E0037, T0009 i luka E0043 pozostaja bez zmian.
