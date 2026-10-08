# Ponowna ocena BTC T−59 — 7 października 2026

## Zakres i pochodzenie granic

Kalendarz obejmuje 84,331 slotów 5-minutowych od pierwszego rekordu serii Gamma (2025-12-18T04:25:00+00:00) do cutoffu startu rynku wyłącznego 2026-10-07T00:00:00+00:00. Gamma potwierdziła 79,680 slotów; pozostałe zachowano jako brak potwierdzonego rekordu.

15 kwietnia 2026 17:05 UTC jest pierwszym rynkiem z przyczynową predykcją po zakończeniu etykiet kalibracyjnych 17:00 UTC. To granica dostępności kalibracji/predykcji, nie początku serii rynków ani BTC. 18 maja 2026 10:30 UTC pochodził z ręcznego `TEST_LAST_MARKET_START` w eksperymencie; nie wynikał z końca rynku ani BTC. Lokalny PMXT kończył się na tym rynku, a nie sam model.

BTC wejściowy jest dostępny od 2026-10-02T18:01:00+00:00 do 2026-10-06T23:59:00+00:00; dopisano 6,119 ciągłych minut. Zamrożony dataset cech kończy się 2 października 18:00 UTC. Model/kalibrator nie były dopasowywane ponownie.

Zamrożony kandydat został wybrany z dwóch chronologicznych foldów 2025 Q1/Q2 i selekcji 2025 Q3. Końcowy fit wariantu `last_3y` używa 1 578 233 etykiet od 1 stycznia 2023 do cutoffu 1 stycznia 2026 (ostatnia dostępność etykiety: 31 grudnia 23:59). Kalibrator używa 30 155 etykiet od 1 stycznia do 15 kwietnia 17:04, ostatnia dostępna 17:00 UTC. Generatory/feature selection mają wspólny boundary fitu 1 stycznia; zamrożone profile stanów kończą się 2 października 18:00.

Predykcje: 50,196; zapisane 9 426 predykcji odtworzono z maksymalną różnicą raw/Platt 0/0. Anchor runtime 2 października ma 0 różnic cech; seria rozszerzona do cutoffu 6 października. Historia jest retrospektywna i nie jest niezależnym holdoutem.

## Pokrycie i luki

Cutoff UTC to 2026-10-07T00:00:00+00:00 dla startów rynków (ostatni slot: 6 października 23:55 UTC). Kalendarz zachowuje niepotwierdzone terminy. Brak potwierdzenia Gamma, etykiety, przyczynowej predykcji, archiwum PMXT i poprawnego asku są osobnymi stanami.

| Miesiąc UTC | sloty | Gamma | wynik | predykcja | snapshot PMXT | UP priceable $5 | DOWN priceable $5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-12 | 3,979 | 3,969 | 3,962 | 0 | 0 | 0 | 0 |
| 2026-01 | 8,928 | 7,475 | 7,469 | 0 | 0 | 0 | 0 |
| 2026-02 | 8,064 | 4,889 | 4,887 | 0 | 0 | 0 | 0 |
| 2026-03 | 8,928 | 8,928 | 8,928 | 0 | 0 | 0 | 0 |
| 2026-04 | 8,640 | 8,636 | 8,634 | 4,403 | 4,401 | 1,375 | 1,694 |
| 2026-05 | 8,928 | 8,927 | 8,917 | 8,928 | 8,927 | 3,764 | 4,017 |
| 2026-06 | 8,640 | 8,634 | 8,634 | 8,640 | 8,634 | 2,377 | 2,355 |
| 2026-07 | 8,928 | 8,927 | 8,927 | 8,928 | 8,927 | 6,072 | 6,064 |
| 2026-08 | 8,928 | 8,928 | 8,927 | 8,928 | 2,593 | 2,338 | 2,333 |
| 2026-09 | 8,640 | 8,639 | 8,639 | 8,640 | 0 | 0 | 0 |
| 2026-10 | 1,728 | 1,728 | 1,728 | 1,728 | 0 | 0 | 0 |

PMXT lokalnie kończył się 18 maja. Po ponownym wykorzystaniu lokalnych partycji, selektywnym pobraniu brakujących historii 17 dodatkowych ID z Gamma oraz historii nowych ID dla lokalnych godzin nakładających się na pierwsze 25 godzin rozszerzenia, archiwum wykonania kończy się na rynku 10 sierpnia 00:00 (wejście 9 sierpnia 23:59:01). Nie pobierano ponownie pełnych lokalnych godzin: zachowano je i rozszerzono tylko o brakujące ID. Dla późniejszych slotów mamy BTC i predykcje, ale brak PMXT T−59; Kacho zaczyna próbki po starcie rynku i nie dostarcza ofert sprzed startu.

Najdłuższe luki ciągłe: `{"market_confirmed": {"longest_true": {"value": true, "slots": 17994, "start_utc": "2026-02-12T00:35:00+00:00", "end_utc": "2026-04-15T12:00:00+00:00"}, "longest_false": {"value": false, "slots": 4622, "start_utc": "2026-01-26T23:25:00+00:00", "end_utc": "2026-02-12T00:30:00+00:00"}, "number_of_true_runs": 15, "number_of_false_runs": 14}, "outcome_available": {"longest_true": {"value": true, "slots": 13714, "start_utc": "2026-02-25T21:15:00+00:00", "end_utc": "2026-04-14T12:00:00+00:00"}, "longest_false": {"value": false, "slots": 4622, "start_utc": "2026-01-26T23:25:00+00:00", "end_utc": "2026-02-12T00:30:00+00:00"}, "number_of_true_runs": 26, "number_of_false_runs": 25}, "btc_input_available": {"longest_true": {"value": true, "slots": 84331, "start_utc": "2025-12-18T04:25:00+00:00", "end_utc": "2026-10-06T23:55:00+00:00"}, "longest_false": null, "number_of_true_runs": 1, "number_of_false_runs": 0}, "causal_prediction_available": {"longest_true": {"value": true, "slots": 50195, "start_utc": "2026-04-15T17:05:00+00:00", "end_utc": "2026-10-06T23:55:00+00:00"}, "longest_false": {"value": false, "slots": 34136, "start_utc": "2025-12-18T04:25:00+00:00", "end_utc": "2026-04-15T17:00:00+00:00"}, "number_of_true_runs": 1, "number_of_false_runs": 1}, "t59_snapshot_available": {"longest_true": {"value": true, "slots": 11444, "start_utc": "2026-04-16T23:40:00+00:00", "end_utc": "2026-05-26T17:15:00+00:00"}, "longest_false": {"value": false, "slots": 34136, "start_utc": "2025-12-18T04:25:00+00:00", "end_utc": "2026-04-15T17:00:00+00:00"}, "number_of_true_runs": 7, "number_of_false_runs": 8}, "ask_data_status_counts": {"outside_causal_prediction_window": 29496, "pmxt_events_observed_by_t59": 26017, "outside_pmxt_archive_window": 16702, "no_pmxt_events_observed_by_t59": 7465, "no_gamma_market_record": 4651}, "expected_market_start_calendar": {"first_market_start_utc": "2025-12-18T04:25:00+00:00", "last_market_start_utc": "2026-10-06T23:55:00+00:00", "market_start_cutoff_exclusive_utc": "2026-10-07T00:00:00+00:00", "slots": 84331, "confirmed_gamma_market_records": 79680}}`

## Diagnoza odrzuceń booka

Stary T−59 cache ma 9,407 rynków. `quote_valid=False` dotyczy 4,838; spośród nich 4,830 miało dwa zainicjalizowane snapshoty — to zgłoszone 4 830. Stare snapshoty pokazywały crossing UP/DOWN odpowiednio w 4,810/4,445 przypadkach; stare rozbieżności zgłoszonego i zrekonstruowanego asku występowały wg rozkładu `{'0': 76, '1': 3312, '2': 1442}`.

Nowy replay umożliwia wycenę $5 po co najmniej jednej natywnej stronie ask w 2,283 z dawnych 4 830 odrzuconych. Spośród nich 0 mają potwierdzony sygnał naprawy rekonstrukcji na tej samej stronie (stara rozbieżność asku znika i nowy BBO jest porównywalny), 2,283 odzyskano przez ocenę asku niezależnie od bidu, a 0 pozostają z niejednoznaczną atrybucją starej rozbieżności. Pozostałe nie są zaliczane do odzyskanych. Szczegóły per token są w `ask_rejection_reasons.csv`; ślady surowych zdarzeń: `reports/btc_preopen/t59_reassessment_20261007/t59_representative_book_traces_20261007.json`.

Semantyka użyta w replayu: `book` zastępuje oba słowniki poziomów, także gdy jedna strona jest pusta; `price_change` BUY aktualizuje bid, SELL ask; ilość jest stanem poziomu, a zero usuwa poziom. Niezależne poziomy zmieniane w jednej grupie timestampu nie są odrzucane; sprzeczne zmiany tego samego poziomu lub niezgodność z równoczesnym snapshotem oznaczają niejednoznaczność tej strony. Osobne grupy odbioru zachowują kolejność receive. Zmiana asku nie jest odrzucana tylko dlatego, że nowszy bid ma timestamp późniejszy. Przeciwny token nie tworzy syntetycznej oferty. Struktura i pola `book`/`price_change` są udokumentowane w [Polymarket Real-Time Data](https://docs.polymarket.com/market-data/realtime-data); usuwanie poziomu przy `size: 0` opisuje też [referencja WebSocket Polymarket](https://github.com/Polymarket/agent-skills/blob/main/websocket.md). Metadane serii sprawdzono przez [Gamma API](https://gamma-api.polymarket.com/docs).

**Klasy przyczyn:** utrata aktualizacji przez inicjalizację/wspólny zegar tokena — błąd implementacji naprawiony; brak źródłowych zdarzeń albo pusty natywny ask — brak danych/oferty; konflikt kolejności o równych timestampach — niejednoznaczne, wyłączone; niewystarczająca głębokość, minimum, limit gotówki lub opłata — zakup niewykonalny dla żądanej stawki.

## Portfele ciągłe

Wszystkie warianty zaczynają od $100 i rozliczają po `max(closedTime, start+5 min)+60 s`. Drawdown liczy equity jako gotówka + pierwotny koszt brutto pozycji oczekujących na rozliczenie; brak mark-to-market. EV jest liczone z rzeczywistego przejścia po askach i opłatach. Nie zmniejszamy stawki przy braku głębokości.

| Polityka | transakcje | obrót | opłaty | końcowy kapitał | PnL | max DD | max koszt pozycji | min gotówka |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_5_usd | 12,671 | $63,355.00 | $26.50 | $1,191.40 | $1,091.40 | 63.63% | $20.00 | $36.00 |
| free_cash_5pct | 7,510 | $437,637.10 | $20.97 | $25.80 | $-74.20 | 99.52% | $524.31 | $25.80 |
| cost_basis_equity_5pct | 7,441 | $350,318.61 | $20.99 | $26.26 | $-73.74 | 99.40% | $416.81 | $26.26 |
| free_cash_5pct_cap20_control | 12,486 | $245,386.43 | $20.97 | $3,972.40 | $3,872.40 | 52.88% | $80.00 | $50.01 |

### PnL miesięczny tego samego portfela

| Polityka | miesiąc | equity otwarcia | equity zamknięcia | PnL |
|---|---|---:|---:|---:|
| fixed_5_usd | 2026-04 | $100.00 | $244.42 | $144.42 |
| fixed_5_usd | 2026-05 | $244.42 | $899.47 | $655.05 |
| fixed_5_usd | 2026-06 | $899.47 | $1,123.81 | $224.33 |
| fixed_5_usd | 2026-07 | $1,123.81 | $1,266.15 | $142.34 |
| fixed_5_usd | 2026-08 | $1,266.15 | $1,191.40 | $-74.74 |
| free_cash_5pct | 2026-04 | $100.00 | $270.10 | $170.10 |
| free_cash_5pct | 2026-05 | $270.10 | $743.88 | $473.78 |
| free_cash_5pct | 2026-06 | $743.88 | $2,062.20 | $1,318.32 |
| free_cash_5pct | 2026-07 | $2,062.20 | $33.34 | $-2,028.86 |
| free_cash_5pct | 2026-08 | $33.34 | $25.80 | $-7.54 |
| cost_basis_equity_5pct | 2026-04 | $100.00 | $271.98 | $171.98 |
| cost_basis_equity_5pct | 2026-05 | $271.98 | $691.74 | $419.77 |
| cost_basis_equity_5pct | 2026-06 | $691.74 | $856.92 | $165.18 |
| cost_basis_equity_5pct | 2026-07 | $856.92 | $27.65 | $-829.28 |
| cost_basis_equity_5pct | 2026-08 | $27.65 | $26.26 | $-1.38 |
| free_cash_5pct_cap20_control | 2026-04 | $100.00 | $270.10 | $170.10 |
| free_cash_5pct_cap20_control | 2026-05 | $270.10 | $2,604.96 | $2,334.87 |
| free_cash_5pct_cap20_control | 2026-06 | $2,604.96 | $3,641.61 | $1,036.65 |
| free_cash_5pct_cap20_control | 2026-07 | $3,641.61 | $4,349.88 | $708.27 |
| free_cash_5pct_cap20_control | 2026-08 | $4,349.88 | $3,972.40 | $-377.48 |

Dzienny log-growth obejmuje startowe $100 do zamknięcia pierwszego dnia, wspólną siatkę dni i końcowe rozliczenie. `daily_log_growth_identity_verified` sprawdza sumę dziennych log-zwrotów względem `log(final/100)`; ruina pozostaje `-Infinity`.

Zamrożony wybór market/side/time dawnej reguły $5, niezależnie przeliczony po nowym asku: `{"status": "independent_bet_diagnostic_not_a_shared_cash_portfolio", "population": "legacy markets, side, and T-59 decision fixed from the old exact-$5 chooser; re-priced on the corrected native ask ladder", "legacy_positive_ev_market_side_times": 2316, "corrected_fixed_side_priceable": 1314, "corrected_fixed_side_unpriceable_reasons": {"ambiguous_equal_source_timestamp_order": 523, "native_ask_bbo_reconciliation_mismatch": 479}, "corrected_fixed_side_sum_pnl_usd": 342.8121426410044, "corrected_fixed_side_mean_pnl_usd": 0.2608920415837172}`. To diagnostyka pojedynczych niezależnych wejść, nie wynik jednego reinwestującego portfela.

## Artefakty i odtwarzalność

Konfiguracja: `configs/research/btc_preopen_t59_reassessment_20261007.json`. Manifest źródeł i sum kontrolnych: `reports/btc_preopen/t59_reassessment_20261007/data_manifest.json`. Kalendarz, drabinki ask, decyzje, transakcje i pokrycie dzienne/miesięczne są zapisane według ścieżek z manifestu poza Git; raport, manifest i konfiguracja pozostają małe i wersjonowalne.

## Ograniczenia

- PMXT to historyczny snapshot archiwum; aktualne `orderMinSize` z Gamma jest używane tylko jako bieżące metadata, a przy braku przyjęto 5 udziałów. Nie odtwarza historycznych minimów.
- Modelowe predykcje przed kalibracyjną granicą 15 kwietnia są celowo puste. Nie zastępujemy ich innym modelem ani OOF z niezgodnym zadaniem.
- PMXT T−59 nie jest dostępny po 9 sierpnia 23:00; wzrost po tej dacie nie jest mierzalny strategią wykonania. Szeroki kalendarz pokazuje tę lukę.
- Book BBO i opłaty odzwierciedlają dostępne archiwalne zdarzenia oraz zachowane założenia fee; rozliczenie gotówki używa `closedTime` jako proxy oficjalnej dostępności wyniku.
- To retrospektywna ocena w obrębie wcześniej używanej historii projektu, nie dowód przewagi poza próbą. Handel pozostaje nieaktywny.

Czas tego przebiegu: 113.2 s; Gamma: 509,325,465 B; Binance: 1,545,572 B; PMXT: 275,826,343 zdarzeń w wybranych wierszach. PMXT transfer: estymacja z reprezentatywnej próbki około 185 GB; licznik dokładnych bajtów HTTP nie był dostępny w pierwotnym extractorze.
