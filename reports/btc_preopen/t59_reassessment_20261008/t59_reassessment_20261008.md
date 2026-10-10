# Weryfikacja replayu BTC T−59 — 8 października 2026

## Wnioski

| Pytanie | Wynik |
|---|---|
| Czy historyczny cache był zgodny? | Nie można tego potwierdzić. SHA snapshotu potwierdza integralność pliku, a nie jego pochodzenie. Stary manifest miał różne fingerprinty archiwum i nie zawierał perplikowych sum treści, mapy wejść ani identyfikatora logiki. Cache odbudowano. |
| Czy dawny wynik +$3,872.40 został potwierdzony? | Nie. To wynik historyczny bez potwierdzonego pochodzenia wejść; poniższe wyniki pochodzą z nowego pełnego replayu. |
| Ile slotów obejmuje kalendarz? | 84,331 kolejnych slotów 5-minutowych; Gamma potwierdziła 79,680, a 50,184 łączy potwierdzony rynek i przyczynową predykcję. |
| Jak daleko sięga wykonawcze archiwum? | T−59 oceniono do cutoffu 2026-10-07T00:00:00+00:00; lokalnie rozszerzone zdarzenia obejmują rynki po PMXT dzięki AG6/V3, z luką pełnego venue opisaną poniżej. |
| Ile dawnych 4,830 odrzuceń odzyskano? | 2,283 z 4,830 ma obecnie wykonalną wycenę asku $5; sygnał zmiany rekonstrukcji: 0, wyjaśnienie kwalifikacją ask-side: 2,283. Pochodzenie starego cache’u pozostaje niezweryfikowane. |

## Granice czasu i wejścia

Kalendarz: 84,331 slotów od 2025-12-18T04:25:00+00:00 (2025-12-18T05:25:00+01:00 Europe/Warsaw) do 2026-10-06T23:55:00+00:00 (2026-10-07T01:55:00+02:00 Europe/Warsaw). Cutoff startu rynku jest wyłączny: 2026-10-07T00:00:00+00:00 (2026-10-07T02:00:00+02:00 Europe/Warsaw). Ostatni slot to 6 października 23:55 UTC / 7 października 01:55 Europe/Warsaw.

Dane Gamma (cc314da279bddf5df4204cc0ff607091f85a985136ac7e7edd1308abcf1b65ba), predykcje (ce6b4439ab689636e506bbc8628ec437211336cc7b2ef53112bc8264c66b01da) i rozszerzenie BTC (5feec1a328047fc1527247aef153e24f22c0f7e670935a23400df5c1f26ffeab) zostały wczytane z lokalnych, wcześniej zapisanych plików i zweryfikowane ich hashami. Zapisany kalendarz odtworzył się dokładnie. Nie pobierano ponownie Gamma/Binance, nie uruchamiano inferencji ani fitu modelu.

Zamrożony model i kalibrator mają SHA256: `{"configs/runtime/btc_preopen_candidate_model_meta.json": "c6b8368fab9c988386e40c36f9b1b8e94a8ed780815a0e14bd297c0b63b24fbd", "data/models/BTC/btc_preopen_candidate_20261005/candidate_calibrator.json": "de2ec02fddae11ca0fed2b0cdb12c2f1842f57a3eb534eddaff8640bc536109e", "data/models/BTC/btc_preopen_candidate_20261005/candidate_model.txt": "19be11d61db42e32aa94c294b516d5ca615cde7377375f5eb265e57423943986"}`. Wykonanie zleceń pozostaje wyłączone.

## Pochodzenie replayu i cache

Stary manifest miał `archive_fingerprint_sha256=1c2895b4df6efbdd82bbf083c63a97c9b4b9e4038e712f6ddd2c7d34b2d994bc` oraz `current_archive_fingerprint_sha256=22ff1b0a7407211fb789dabf7918f5b0dc56fad1c2be148649aedced065c6b7c`. Bieżący fingerprint obliczony tą samą starą regułą wynosi `22ff1b0a7407211fb789dabf7918f5b0dc56fad1c2be148649aedced065c6b7c` i zgadza się z zapisanym polem current (`True`). Wersja z commita `d7557bc` hashowała nazwy, rozmiary i `mtime_ns`; różnica 1c2895…→22ff1b… dowodzi zmiany agregatu metadanych, ale historyczne per-file rekordy tych pól nie przetrwały, więc nie da się wskazać nazw ani rodzaju zmiany metadanych dla poszczególnych plików. Nie istniał historyczny fingerprint treści, więc nie można rozdzielić zmian metadanych od zmian danych w wejściach starego snapshotu. Obecne 2,813 partycji PMXT zgadza się z hashami ekstrakcyjnego checkpointu; to potwierdza obecne pliki względem checkpointu ekstraktora, ale nie dowodzi pochodzenia starego snapshotu.

Nowy replay manifest v3 wiąże SHA treści i schematu każdej partycji, indeks rynków, mapę tokenów, konfigurację wejścia oraz hash kodu rekonstrukcji. Snapshot SHA `c90bb2364fff225b6e5047ff761f5ea150dda35afdd63a88ed0fef6f6bd6c941` potwierdza integralność artefaktu; `replay_provenance_verified=True` potwierdza zgodność zależności. Poprzedni cache pozostaje w `data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261007/historical_unverified_20261007/`.

## Archiwa i rzeczywista dostępność

| Źródło | Opublikowane godziny | Przetworzone godziny | Zdarzenia przetworzone | Zasięg / status |
|---|---:|---:|---:|---|
| PMXT V2 | 2,835 | 2,813 | 275,826,343 | 2026-04-13T19:00:00+00:00 .. 2026-08-10T00:00:00+00:00; existing local output reused; missing index hours=3 |
| AG6 V2 | 136 | 134 | 16,021,217 | 2026-08-01T13:00:00+00:00 .. 2026-08-15T09:00:00+00:00; sparse mirror; missing within span=197 |
| V3 | 1,194 | 1,194 | 276,978,907 | 2026-08-18T06:00:00+00:00 .. 2026-10-06T23:00:00+00:00; selected row groups only; missing within span=0 |
| full venue archive gap | 0 | 0 | 0 | 68 hours: 2026-08-15T10:00:00+00:00 .. 2026-08-18T05:00:00+00:00 |

PMXT V2 checksum index publikuje zakres 2026-04-13T19:00:00+00:00–2026-08-10T00:00:00+00:00; lokalne wyekstrahowane partycje obejmują 2,813 plików. W indeksie są trzy brakujące godziny: `['2026-06-11T04:00:00+00:00', '2026-06-11T05:00:00+00:00', '2026-06-11T06:00:00+00:00']`. Sprawdzono indeks PMXT [PMXT V2](https://archive.pendulumflow.com/pmxt/v2/?page=1).
AG6 publikuje 136 godzin w zakresie 2026-08-01T13:00:00+00:00–2026-08-15T09:00:00+00:00; indeks pokrywa tylko część godzin tego przedziału. Wydawca podaje brak warunków licencji AG6, dlatego dane i partycje pochodne pozostają lokalne. [Indeks AG6](https://archive.pendulumflow.com/third-party/ag6/), [opis formatu i licencji](https://archive.pendulumflow.com/formats/ag6).
V3 publikuje 1,242 plików w zasięgu docelowym od 2026-08-18T06:00:00+00:00 do 2026-10-06T23:00:00+00:00; pobrano selektywnie wymagane grupy wierszy, bez pełnych plików godzinowych. [Indeks V3 i sumy kontrolne](https://archive.pendulumflow.com/v3/).
Między końcem AG6 i początkiem V3 indeks nie pokazuje pełnego archiwum venue przez 68 godzin (2026-08-15T10:00:00+00:00–2026-08-18T05:00:00+00:00). Zapisano 0 nierozwiązanych prób, klasy: `{}`. HTTP 404 dla obiektu obecnego w checksum index oznacza `advertised_source_object_missing`, HTTP 429 oznacza `access_rate_limited`; godziny bez wpisu w indeksie są raportowane oddzielnie. Każda próba ma godzinę, URL, status HTTP i błąd w lokalnym `archive_manifest.json`. Sumy obiektów źródłowych są deklarowane przez indeks, a pełnych zdalnych plików nie hashowano lokalnie — hashowano znormalizowane wyjścia i scalone partycje.

## Pokrycie kalendarza

| Miesiąc UTC | sloty | Gamma | predykcja przyczynowa | godzina wejścia w archiwum | kompletne 25h | zdarzenia T−59 | oba booki | UP $5 | DOWN $5 | zakup $5 z dodatnim EV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-12 | 3,979 | 3,969 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2026-01 | 8,928 | 7,475 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2026-02 | 8,064 | 4,889 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2026-03 | 8,928 | 8,928 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2026-04 | 8,640 | 8,636 | 4,403 | 4,401 | 4,401 | 4,401 | 4,401 | 1,375 | 1,694 | 304 |
| 2026-05 | 8,928 | 8,927 | 8,928 | 8,927 | 8,927 | 8,609 | 8,608 | 3,764 | 4,017 | 3,612 |
| 2026-06 | 8,640 | 8,634 | 8,640 | 8,598 | 8,299 | 3,588 | 3,588 | 2,377 | 2,355 | 1,900 |
| 2026-07 | 8,928 | 8,927 | 8,928 | 8,927 | 8,927 | 6,826 | 6,826 | 6,072 | 6,064 | 4,925 |
| 2026-08 | 8,928 | 8,928 | 8,928 | 8,112 | 7,813 | 8,108 | 8,012 | 5,463 | 5,748 | 4,371 |
| 2026-09 | 8,640 | 8,639 | 8,640 | 8,639 | 8,639 | 8,639 | 8,639 | 6,359 | 7,512 | 5,316 |
| 2026-10 | 1,728 | 1,728 | 1,728 | 1,728 | 1,728 | 1,728 | 1,728 | 1,722 | 1,723 | 1,301 |

Pokrycie zgrupowano przez połączony zbiór godzin PMXT+AG6+V3. Dla każdego slotu kalendarz lokalny zachowuje Gamma, predykcję, dostępność godziny wejścia, liczbę źródłowych/przetworzonych godzin w oknie 25h, zdarzenia T−59, inicjalizację obu ksiąg, wycenę UP/DOWN dla $5, wykonalny zakup i wynik/skip każdej polityki. Brak dodatniego EV pozostaje osobnym powodem od braku danych, braku native asku, głębokości, minimum i gotówki.

Najdłuższe ciągłe przebiegi: `{"market_confirmed": {"longest_true": {"value": true, "slots": 17994, "start_utc": "2026-02-12T00:35:00+00:00", "end_utc": "2026-04-15T12:00:00+00:00"}, "longest_false": {"value": false, "slots": 4622, "start_utc": "2026-01-26T23:25:00+00:00", "end_utc": "2026-02-12T00:30:00+00:00"}, "number_of_true_runs": 15, "number_of_false_runs": 14}, "outcome_available": {"longest_true": {"value": true, "slots": 13714, "start_utc": "2026-02-25T21:15:00+00:00", "end_utc": "2026-04-14T12:00:00+00:00"}, "longest_false": {"value": false, "slots": 4622, "start_utc": "2026-01-26T23:25:00+00:00", "end_utc": "2026-02-12T00:30:00+00:00"}, "number_of_true_runs": 26, "number_of_false_runs": 25}, "btc_input_available": {"longest_true": {"value": true, "slots": 84331, "start_utc": "2025-12-18T04:25:00+00:00", "end_utc": "2026-10-06T23:55:00+00:00"}, "longest_false": null, "number_of_true_runs": 1, "number_of_false_runs": 0}, "causal_prediction_available": {"longest_true": {"value": true, "slots": 50195, "start_utc": "2026-04-15T17:05:00+00:00", "end_utc": "2026-10-06T23:55:00+00:00"}, "longest_false": {"value": false, "slots": 34136, "start_utc": "2025-12-18T04:25:00+00:00", "end_utc": "2026-04-15T17:00:00+00:00"}, "number_of_true_runs": 1, "number_of_false_runs": 1}, "archive_entry_hour_available": {"longest_true": {"value": true, "slots": 11444, "start_utc": "2026-04-16T23:40:00+00:00", "end_utc": "2026-05-26T17:15:00+00:00"}, "longest_false": {"value": false, "slots": 34136, "start_utc": "2025-12-18T04:25:00+00:00", "end_utc": "2026-04-15T17:00:00+00:00"}, "number_of_true_runs": 10, "number_of_false_runs": 10}, "archive_history_complete": {"longest_true": {"value": true, "slots": 11444, "start_utc": "2026-04-16T23:40:00+00:00", "end_utc": "2026-05-26T17:15:00+00:00"}, "longest_false": {"value": false, "slots": 34136, "start_utc": "2025-12-18T04:25:00+00:00", "end_utc": "2026-04-15T17:00:00+00:00"}, "number_of_true_runs": 10, "number_of_false_runs": 10}, "archive_freshness_window_available": {"longest_true": {"value": true, "slots": 11444, "start_utc": "2026-04-16T23:40:00+00:00", "end_utc": "2026-05-26T17:15:00+00:00"}, "longest_false": {"value": false, "slots": 34136, "start_utc": "2025-12-18T04:25:00+00:00", "end_utc": "2026-04-15T17:00:00+00:00"}, "number_of_true_runs": 10, "number_of_false_runs": 10}, "t59_snapshot_available": {"longest_true": {"value": true, "slots": 15418, "start_utc": "2026-07-12T02:30:00+00:00", "end_utc": "2026-09-03T15:15:00+00:00"}, "longest_false": {"value": false, "slots": 34136, "start_utc": "2025-12-18T04:25:00+00:00", "end_utc": "2026-04-15T17:00:00+00:00"}, "number_of_true_runs": 8, "number_of_false_runs": 8}, "t59_any_event_observed": {"longest_true": {"value": true, "slots": 9607, "start_utc": "2026-09-03T15:25:00+00:00", "end_utc": "2026-10-06T23:55:00+00:00"}, "longest_false": {"value": false, "slots": 34136, "start_utc": "2025-12-18T04:25:00+00:00", "end_utc": "2026-04-15T17:00:00+00:00"}, "number_of_true_runs": 32, "number_of_false_runs": 32}, "t59_both_books_initialized": {"longest_true": {"value": true, "slots": 9607, "start_utc": "2026-09-03T15:25:00+00:00", "end_utc": "2026-10-06T23:55:00+00:00"}, "longest_false": {"value": false, "slots": 34136, "start_utc": "2025-12-18T04:25:00+00:00", "end_utc": "2026-04-15T17:00:00+00:00"}, "number_of_true_runs": 75, "number_of_false_runs": 75}, "fixed5_buy_eligible": {"longest_true": {"value": true, "slots": 45, "start_utc": "2026-07-15T21:30:00+00:00", "end_utc": "2026-07-16T01:10:00+00:00"}, "longest_false": {"value": false, "slots": 34161, "start_utc": "2025-12-18T04:25:00+00:00", "end_utc": "2026-04-15T19:05:00+00:00"}, "number_of_true_runs": 6938, "number_of_false_runs": 6938}, "ask_data_status_counts": {"archive_events_observed_by_t59": 41863, "outside_causal_prediction_window": 29496, "no_archive_events_observed_by_t59": 7468, "no_gamma_market_record": 4651, "no_full_venue_archive_for_period": 816, "archive_hour_missing_at_entry": 37}, "expected_market_start_calendar": {"first_market_start_utc": "2025-12-18T04:25:00+00:00", "last_market_start_utc": "2026-10-06T23:55:00+00:00", "market_start_cutoff_exclusive_utc": "2026-10-07T00:00:00+00:00", "slots": 84331, "confirmed_gamma_market_records": 79680}}`. Pełny kalendarz jest lokalnie w `coverage_calendar.parquet`; małe miesięczne podsumowanie jest w `monthly_coverage.csv`.

## Wyniki czterech portfeli

Wartość kapitału to gotówka plus pierwotny koszt brutto pozycji nierozliczonych. Zakup obciąża gotówkę jeden raz; koszt brutto blokuje kapitał, a gotówka wraca dopiero po wypłacie. EV używa netto otrzymanych udziałów i faktycznego debetu gotówkowego. PnL / dolar wydany w tabeli miesięcznej to PnL kohorty zakupów miesiąca podzielony przez sumę faktycznie pobranej gotówki wraz z opłatą gotówkową.

| Polityka | transakcje | obrót | opłaty | końcowa gotówka | PnL | max DD | przedział max DD | min wolna gotówka | audyt |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| fixed_5_usd | 21,726 | $108,630.00 | $26.50 | $1,724.99 | $1,624.99 | 63.63% | 2026-04-18T06:51:20+00:00 → 2026-04-25T07:46:18+00:00 | $36.00 | passed |
| free_cash_5pct | 7,510 | $437,637.10 | $20.97 | $25.80 | $-74.20 | 99.52% | 2026-06-09T22:31:21+00:00 → 2026-08-08T21:32:25+00:00 | $25.80 | passed |
| cost_basis_equity_5pct | 7,442 | $350,319.93 | $20.99 | $24.95 | $-75.05 | 99.43% | 2026-05-26T18:21:25+00:00 → 2026-08-29T14:11:54+00:00 | $24.95 | passed |
| free_cash_5pct_cap20_control | 21,145 | $418,566.43 | $20.97 | $6,053.50 | $5,953.50 | 52.88% | 2026-04-18T06:51:20+00:00 → 2026-04-25T07:46:18+00:00 | $50.01 | passed |

### Wyniki miesięczne

| Polityka | miesiąc UTC | equity start | equity koniec | PnL equity | transakcje | obrót | gotówka wydana | PnL kohorty / $ | mediana stawki | max DD miesiąca | pominięcia |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fixed_5_usd | 2026-04 | $100.00 | $244.42 | $144.42 | 304 | $1,520.00 | $1,520.00 | 0.095 | $5.00 | 63.63% | `{"no_positive_net_expected_value": 2294, "no_priceable_native_ask_side": 1804}` |
| fixed_5_usd | 2026-05 | $244.42 | $899.47 | $655.05 | 3,611 | $18,055.00 | $18,055.00 | 0.037 | $5.00 | 26.99% | `{"missing_official_outcome_or_settlement_time": 10, "no_positive_net_expected_value": 1840, "no_priceable_native_ask_side": 3466}` |
| fixed_5_usd | 2026-06 | $899.47 | $1,123.81 | $224.33 | 1,899 | $9,495.00 | $9,495.00 | 0.023 | $5.00 | 14.86% | `{"no_positive_net_expected_value": 550, "no_priceable_native_ask_side": 6185}` |
| fixed_5_usd | 2026-07 | $1,123.81 | $1,266.15 | $142.34 | 4,926 | $24,630.00 | $24,630.00 | 0.006 | $5.00 | 33.17% | `{"no_positive_net_expected_value": 1521, "no_priceable_native_ask_side": 2480}` |
| fixed_5_usd | 2026-08 | $1,266.15 | $1,551.32 | $285.17 | 4,369 | $21,845.00 | $21,845.00 | 0.013 | $5.00 | 21.17% | `{"missing_official_outcome_or_settlement_time": 1, "no_positive_net_expected_value": 1942, "no_priceable_native_ask_side": 2616}` |
| fixed_5_usd | 2026-09 | $1,551.32 | $2,087.13 | $535.82 | 5,317 | $26,585.00 | $26,585.00 | 0.020 | $5.00 | 12.09% | `{"no_positive_net_expected_value": 3212, "no_priceable_native_ask_side": 110}` |
| fixed_5_usd | 2026-10 | $2,087.13 | $1,724.99 | $-362.14 | 1,300 | $6,500.00 | $6,500.00 | -0.054 | $5.00 | 18.93% | `{"no_positive_net_expected_value": 424, "no_priceable_native_ask_side": 3}` |
| free_cash_5pct | 2026-04 | $100.00 | $270.10 | $170.10 | 299 | $2,456.33 | $2,456.33 | 0.069 | $6.83 | 52.88% | `{"no_positive_net_expected_value": 2299, "no_priceable_native_ask_side": 1804}` |
| free_cash_5pct | 2026-05 | $270.10 | $743.88 | $473.78 | 3,339 | $211,583.41 | $211,583.41 | 0.003 | $44.65 | 89.59% | `{"missing_official_outcome_or_settlement_time": 10, "no_positive_net_expected_value": 2112, "no_priceable_native_ask_side": 3466}` |
| free_cash_5pct | 2026-06 | $743.88 | $2,062.20 | $1,318.32 | 1,746 | $137,048.82 | $137,048.82 | 0.009 | $63.87 | 77.59% | `{"no_positive_net_expected_value": 703, "no_priceable_native_ask_side": 6185}` |
| free_cash_5pct | 2026-07 | $2,062.20 | $33.34 | $-2,028.86 | 2,116 | $86,533.46 | $86,533.46 | -0.023 | $31.21 | 99.01% | `{"below_minimum_order_shares": 3575, "no_positive_net_expected_value": 756, "no_priceable_native_ask_side": 2480}` |
| free_cash_5pct | 2026-08 | $33.34 | $25.80 | $-7.54 | 10 | $15.08 | $15.08 | -0.500 | $1.50 | 22.62% | `{"below_minimum_order_shares": 6301, "missing_official_outcome_or_settlement_time": 1, "no_priceable_native_ask_side": 2616}` |
| free_cash_5pct | 2026-09 | $25.80 | $25.80 | $0.00 | 0 | $0.00 | $0.00 | — | — | 0.00% | `{"below_minimum_order_shares": 8529, "no_priceable_native_ask_side": 110}` |
| free_cash_5pct | 2026-10 | $25.80 | $25.80 | $0.00 | 0 | $0.00 | $0.00 | — | — | 0.00% | `{"below_minimum_order_shares": 1724, "no_priceable_native_ask_side": 3}` |
| cost_basis_equity_5pct | 2026-04 | $100.00 | $271.98 | $171.98 | 299 | $2,527.00 | $2,527.00 | 0.068 | $7.07 | 53.01% | `{"no_positive_net_expected_value": 2299, "no_priceable_native_ask_side": 1804}` |
| cost_basis_equity_5pct | 2026-05 | $271.98 | $691.74 | $419.77 | 3,335 | $215,708.10 | $215,708.10 | 0.002 | $43.05 | 90.31% | `{"missing_official_outcome_or_settlement_time": 10, "no_positive_net_expected_value": 2116, "no_priceable_native_ask_side": 3466}` |
| cost_basis_equity_5pct | 2026-06 | $691.74 | $856.92 | $165.18 | 1,806 | $101,023.12 | $101,023.12 | 0.001 | $50.18 | 85.00% | `{"no_positive_net_expected_value": 643, "no_priceable_native_ask_side": 6185}` |
| cost_basis_equity_5pct | 2026-07 | $856.92 | $27.65 | $-829.28 | 1,995 | $31,051.33 | $31,051.33 | -0.027 | $13.53 | 97.99% | `{"below_minimum_order_shares": 3761, "no_positive_net_expected_value": 691, "no_priceable_native_ask_side": 2480}` |
| cost_basis_equity_5pct | 2026-08 | $27.65 | $24.95 | $-2.69 | 7 | $10.37 | $10.37 | -0.260 | $1.46 | 26.49% | `{"below_minimum_order_shares": 6304, "missing_official_outcome_or_settlement_time": 1, "no_priceable_native_ask_side": 2616}` |
| cost_basis_equity_5pct | 2026-09 | $24.95 | $24.95 | $0.00 | 0 | $0.00 | $0.00 | — | — | 0.00% | `{"below_minimum_order_shares": 8529, "no_priceable_native_ask_side": 110}` |
| cost_basis_equity_5pct | 2026-10 | $24.95 | $24.95 | $0.00 | 0 | $0.00 | $0.00 | — | — | 0.00% | `{"below_minimum_order_shares": 1724, "no_priceable_native_ask_side": 3}` |
| free_cash_5pct_cap20_control | 2026-04 | $100.00 | $270.10 | $170.10 | 299 | $2,456.33 | $2,456.33 | 0.069 | $6.83 | 52.88% | `{"no_positive_net_expected_value": 2299, "no_priceable_native_ask_side": 1804}` |
| free_cash_5pct_cap20_control | 2026-05 | $270.10 | $2,604.96 | $2,334.87 | 3,534 | $69,870.10 | $69,870.10 | 0.034 | $20.00 | 44.62% | `{"missing_official_outcome_or_settlement_time": 10, "no_positive_net_expected_value": 1917, "no_priceable_native_ask_side": 3466}` |
| free_cash_5pct_cap20_control | 2026-06 | $2,604.96 | $3,641.61 | $1,036.65 | 1,875 | $37,500.00 | $37,500.00 | 0.027 | $20.00 | 18.28% | `{"no_positive_net_expected_value": 574, "no_priceable_native_ask_side": 6185}` |
| free_cash_5pct_cap20_control | 2026-07 | $3,641.61 | $4,349.88 | $708.27 | 4,874 | $97,480.00 | $97,480.00 | 0.007 | $20.00 | 38.20% | `{"no_positive_net_expected_value": 1573, "no_priceable_native_ask_side": 2480}` |
| free_cash_5pct_cap20_control | 2026-08 | $4,349.88 | $5,271.36 | $921.48 | 4,207 | $84,140.00 | $84,140.00 | 0.011 | $20.00 | 24.85% | `{"missing_official_outcome_or_settlement_time": 1, "no_positive_net_expected_value": 2104, "no_priceable_native_ask_side": 2616}` |
| free_cash_5pct_cap20_control | 2026-09 | $5,271.36 | $7,305.44 | $2,034.08 | 5,116 | $102,320.00 | $102,320.00 | 0.019 | $20.00 | 13.58% | `{"no_positive_net_expected_value": 3413, "no_priceable_native_ask_side": 110}` |
| free_cash_5pct_cap20_control | 2026-10 | $7,305.44 | $6,053.50 | $-1,251.94 | 1,240 | $24,800.00 | $24,800.00 | -0.049 | $20.00 | 18.70% | `{"no_positive_net_expected_value": 484, "no_priceable_native_ask_side": 3}` |

## Lipiec: sizing, wybór strony i wykonanie

Porównanie poniżej jest diagnostyczne, nie addytywne: kolejne wygrane/przegrane zmieniają gotówkę, wielkość późniejszych zleceń, wykonalność i terminy wejść. ‘zamrożona strona $5’ utrzymuje stronę wybraną przy $5 i zmienia sizing; ‘strona rzeczywiście wybrana, stawka $5’ kontroluje sizing przy wyborze z dynamicznej stawki.

| Polityka | PnL rzeczywisty | PnL przy zamrożonej stronie $5 i tym sizingu | PnL: dynamicznie wybrana strona ze stawką $5 | wejścia wspólne / ta sama strona | zmiana strony po zmianie stawki | mediana zmiany VWAP vs $5 | opłaty | odrzucenia minimum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cost_basis_equity_5pct | $-829.28 | $-829.28 | $-96.30 | 1,995 / 1,995 | 0/1,995 | $0.00 | $0.00 | 3,761 |
| free_cash_5pct | $-2,028.86 | $-2,029.42 | $-151.52 | 2,115 / 2,115 | 0/2,115 | $0.00 | $0.00 | 3,575 |
| free_cash_5pct_cap20_control | $707.49 | $707.49 | $190.97 | 4,874 / 4,874 | 0/4,874 | $0.00 | $0.00 | 0 |
- `cost_basis_equity_5pct`: rzeczywisty PnL lipca to $-829.28; portfel o tych samych regułach sizingu ze stroną zamrożoną według $5 dał $-829.28, a strona wybrana przez dynamiczną stawkę przy stałych $5 dała $-96.30. Zmiana strony wystąpiła w 0 z 1,995 decyzji, gdzie obie wielkości miały dodatni wybór. Te kontrole nie są addytywną dekompozycją PnL.
- `free_cash_5pct`: rzeczywisty PnL lipca to $-2,028.86; portfel o tych samych regułach sizingu ze stroną zamrożoną według $5 dał $-2,029.42, a strona wybrana przez dynamiczną stawkę przy stałych $5 dała $-151.52. Zmiana strony wystąpiła w 0 z 2,115 decyzji, gdzie obie wielkości miały dodatni wybór. Te kontrole nie są addytywną dekompozycją PnL.
- `free_cash_5pct_cap20_control`: rzeczywisty PnL lipca to $707.49; portfel o tych samych regułach sizingu ze stroną zamrożoną według $5 dał $707.49, a strona wybrana przez dynamiczną stawkę przy stałych $5 dała $190.97. Zmiana strony wystąpiła w 0 z 4,874 decyzji, gdzie obie wielkości miały dodatni wybór. Te kontrole nie są addytywną dekompozycją PnL.

### Największe wkłady w maksymalne obsunięcie

| Polityka | rynek start | strona | stawka brutto | wkład w peak→trough | PnL trade | rozliczenie |
|---|---|---|---:|---:|---:|---|
| fixed_5_usd | 2026-09-22 22:20:00+00:00 | up | $5.00 | $-5.00 | $-5.00 | 2026-09-22 22:26:53+00:00 |
| fixed_5_usd | 2026-09-29 15:25:00+00:00 | up | $5.00 | $-5.00 | $-5.00 | 2026-09-29 15:31:53+00:00 |
| fixed_5_usd | 2026-10-01 05:10:00+00:00 | down | $5.00 | $-5.00 | $-5.00 | 2026-10-01 05:16:54+00:00 |
| fixed_5_usd | 2026-10-01 05:20:00+00:00 | down | $5.00 | $-5.00 | $-5.00 | 2026-10-01 05:26:53+00:00 |
| fixed_5_usd | 2026-10-01 07:25:00+00:00 | up | $5.00 | $-5.00 | $-5.00 | 2026-10-01 07:32:27+00:00 |
| fixed_5_usd | 2026-10-03 12:45:00+00:00 | down | $5.00 | $-5.00 | $-5.00 | 2026-10-03 12:51:53+00:00 |
| fixed_5_usd | 2026-10-05 04:25:00+00:00 | up | $5.00 | $-5.00 | $-5.00 | 2026-10-05 04:31:55+00:00 |
| fixed_5_usd | 2026-10-05 04:35:00+00:00 | up | $5.00 | $-5.00 | $-5.00 | 2026-10-05 04:42:28+00:00 |
| fixed_5_usd | 2026-10-05 15:05:00+00:00 | up | $5.00 | $-5.00 | $-5.00 | 2026-10-05 15:11:53+00:00 |
| fixed_5_usd | 2026-10-06 06:05:00+00:00 | up | $5.00 | $-5.00 | $-5.00 | 2026-10-06 06:11:52+00:00 |
| free_cash_5pct | 2026-06-09 22:50:00+00:00 | up | $268.88 | $-268.88 | $-268.88 | 2026-06-09 22:56:18+00:00 |
| free_cash_5pct | 2026-06-09 22:55:00+00:00 | up | $255.43 | $-255.43 | $-255.43 | 2026-06-09 23:01:18+00:00 |
| free_cash_5pct | 2026-06-09 23:10:00+00:00 | up | $247.90 | $-247.90 | $-247.90 | 2026-06-09 23:16:21+00:00 |
| free_cash_5pct | 2026-06-09 23:30:00+00:00 | up | $235.50 | $-235.50 | $-235.50 | 2026-06-09 23:36:21+00:00 |
| free_cash_5pct | 2026-06-10 01:45:00+00:00 | down | $231.58 | $-231.58 | $-231.58 | 2026-06-10 01:51:25+00:00 |
| free_cash_5pct | 2026-06-09 23:05:00+00:00 | up | $230.53 | $-230.53 | $-230.53 | 2026-06-09 23:11:19+00:00 |
| free_cash_5pct | 2026-06-09 23:45:00+00:00 | up | $223.73 | $-223.73 | $-223.73 | 2026-06-09 23:51:18+00:00 |
| free_cash_5pct | 2026-06-10 00:05:00+00:00 | down | $221.82 | $-221.82 | $-221.82 | 2026-06-10 00:11:21+00:00 |
| free_cash_5pct | 2026-06-10 02:00:00+00:00 | up | $220.00 | $-220.00 | $-220.00 | 2026-06-10 02:06:19+00:00 |
| free_cash_5pct | 2026-06-10 02:10:00+00:00 | up | $209.00 | $-209.00 | $-209.00 | 2026-06-10 02:16:21+00:00 |
| cost_basis_equity_5pct | 2026-05-26 18:20:00+00:00 | down | $208.40 | $-208.40 | $-208.40 | 2026-05-26 18:26:15+00:00 |
| cost_basis_equity_5pct | 2026-05-26 18:35:00+00:00 | up | $208.00 | $-208.00 | $-208.00 | 2026-05-26 18:41:17+00:00 |
| cost_basis_equity_5pct | 2026-05-26 18:40:00+00:00 | up | $208.00 | $-208.00 | $-208.00 | 2026-05-26 18:46:26+00:00 |
| cost_basis_equity_5pct | 2026-05-26 18:45:00+00:00 | down | $197.60 | $-197.60 | $-197.60 | 2026-05-26 18:51:15+00:00 |
| cost_basis_equity_5pct | 2026-05-28 13:50:00+00:00 | down | $193.22 | $-193.22 | $-193.22 | 2026-05-28 13:56:18+00:00 |
| cost_basis_equity_5pct | 2026-05-28 14:50:00+00:00 | down | $191.17 | $-191.17 | $-191.17 | 2026-05-28 14:56:18+00:00 |
| cost_basis_equity_5pct | 2026-05-28 15:25:00+00:00 | down | $191.06 | $-191.06 | $-191.06 | 2026-05-28 15:31:18+00:00 |
| cost_basis_equity_5pct | 2026-05-27 10:15:00+00:00 | down | $190.31 | $-190.31 | $-190.31 | 2026-05-27 10:21:18+00:00 |
| cost_basis_equity_5pct | 2026-05-28 15:35:00+00:00 | down | $190.23 | $-190.23 | $-190.23 | 2026-05-28 15:41:17+00:00 |
| cost_basis_equity_5pct | 2026-05-28 15:40:00+00:00 | down | $190.23 | $-190.23 | $-190.23 | 2026-05-28 15:46:18+00:00 |
| free_cash_5pct_cap20_control | 2026-07-06 13:00:00+00:00 | up | $20.00 | $-20.00 | $-20.00 | 2026-07-06 13:06:20+00:00 |
| free_cash_5pct_cap20_control | 2026-07-08 08:35:00+00:00 | up | $20.00 | $-20.00 | $-20.00 | 2026-07-08 08:41:19+00:00 |
| free_cash_5pct_cap20_control | 2026-07-08 13:10:00+00:00 | up | $20.00 | $-20.00 | $-20.00 | 2026-07-08 13:16:53+00:00 |
| free_cash_5pct_cap20_control | 2026-07-08 13:15:00+00:00 | up | $20.00 | $-20.00 | $-20.00 | 2026-07-08 13:21:19+00:00 |
| free_cash_5pct_cap20_control | 2026-07-16 23:30:00+00:00 | up | $20.00 | $-20.00 | $-20.00 | 2026-07-16 23:36:26+00:00 |
| free_cash_5pct_cap20_control | 2026-07-17 00:35:00+00:00 | up | $20.00 | $-20.00 | $-20.00 | 2026-07-17 00:41:26+00:00 |
| free_cash_5pct_cap20_control | 2026-07-17 11:50:00+00:00 | down | $20.00 | $-20.00 | $-20.00 | 2026-07-17 11:56:22+00:00 |
| free_cash_5pct_cap20_control | 2026-07-20 06:40:00+00:00 | up | $20.00 | $-20.00 | $-20.00 | 2026-07-20 06:46:21+00:00 |
| free_cash_5pct_cap20_control | 2026-07-22 05:20:00+00:00 | up | $20.00 | $-20.00 | $-20.00 | 2026-07-22 05:26:16+00:00 |
| free_cash_5pct_cap20_control | 2026-07-23 01:20:00+00:00 | up | $20.00 | $-20.00 | $-20.00 | 2026-07-23 01:26:25+00:00 |

### Minimalny rozmiar zlecenia

| Polityka | pominięcia przez minimum | pierwsze UTC | gotówka przed | żądana stawka | minimum brutto udziałów | źródło |
|---|---:|---|---:|---:|---:|---|
| fixed_5_usd | 0 | None | — | — | None | None |
| free_cash_5pct | 20,129 | 2026-07-18T18:24:01+00:00 | $49.18 | $2.46 | 5.0 | current_gamma_metadata |
| cost_basis_equity_5pct | 20,318 | 2026-07-18T04:19:01+00:00 | $46.83 | $2.47 | 5.0 | current_gamma_metadata |
| free_cash_5pct_cap20_control | 0 | None | — | — | None | None |

Decyzje zachowują obie strony przy $5 i przy stawce żądanej przez portfel: EV w dolarach, VWAP, udziały brutto/netto, opłatę, debet, minimalne udziały i przyczynę niewykonalności. Lokalne `sizing_opportunities.parquet` pozwala sprawdzić wspólne wejścia i wejścia wykonalne tylko dla części polityk. `drawdown_contributors.csv` pokazuje transakcje z największym ujemnym wkładem w rzeczywistym przedziale peak→trough; wkład sumuje opłatę gotówkową przy wejściu i payout minus zablokowany koszt przy rozliczeniu.

Najwcześniejsze pominięcie przez minimalny rozmiar, według bieżącego `orderMinSize` z Gamma lub fallbacku 5 udziałów, zapisano per polityka w `july_policy_attribution.csv`; wartości nie są historycznym odczytem minimów. Minimalny rozmiar jest w udziałach brutto, przed potrąceniem opłaty udziałowej; zakupione udziały netto zapisano osobno.

## Dawne 4,830 odrzuceń

Pierwotny zestaw obejmuje 9,407 rynków; 4,830 miało dwa zainicjalizowane booki, ale `quote_valid=false`. W nowym replayu wykonalny ask $5 ma 2,283 z nich. Dla 0 nie stwierdzono sygnału naprawy rekonstrukcji po porównaniu BBO; 2,283 przechodzi obecną ocenę ask-side mimo dawnego odrzucenia połączonego quote. Pozostałe 2,547 nie mają obecnie wykonalnej wyceny $5. Poprzednia diagnoza podała ten sam wynik 2,283. Nie dowodzi to, że stary kod był poprawny ani błędny: historyczny cache nie ma perplikowych hashy treści ani identyfikatora logiki.

To inna kohorta niż dodatkowy plik `historical_unverified_20261007/t59_ask_ladders.parquet`: ma on 5,817 pełnych snapshotów z `quote_valid=false`, z czego 2,439 mają obecnie ask $5; wszystkie należą do klasy `2,439`. `old_replay_provenance_verified=false`; tych 5,817 snapshotów nie należy utożsamiać z pierwotnymi 4,830.

Porównanie per market dla tej dodatkowej kohorty zachowuje dawny i nowy best ask, VWAP, ilość brutto/netto, status obu stron i powód odrzucenia w lokalnym `legacy_recovery_comparison.parquet`. Klasy odzyskania i limity atrybucji są w `legacy_recovery_classes.csv`.

## Wydajność i odzyskanie checkpointu

Replay użył 16 workerów. Zweryfikowany checkpoint po partycji 3,420 zachował ten prefiks; w kontynuacji przetworzono 717 z 4,137 partycji, bez ponownego replayu ukończonego prefiksu. Partie kontynuacji zawierały 201,051,794 wierszy źródłowych (20,252 wierszy/s) i zajęły 9927.5s.
Na tym samym teście z 95,406 zdarzeniami i 11 rynkami serial trwał 34.01s, a równoległy replay z 11 workerami 10.15s (3.35×); snapshoty były identyczne. Sprzęt: i7-13650HX (14 rdzeni/20 wątków), 63.7 GiB RAM; GPU nie użyto. Przy checkpointcie 4,032 łączny RSS procesu i workerów wynosił ok. 11 GiB, a system miał 37.8 GiB wolnej RAM; to próbka, nie pomiar szczytowy. Całkowite obciążenie CPU w próbkach wynosiło 79–96% przy równoległym lokalnym workloadzie, którego nie przerywano.
Szczegółowe czasy odczytu, konwersji/filtrowania, sortowania, pętli grupującej i aktualizacji booków oraz zapisu checkpointu są w `replay_performance_profile.md`.

## Rachunkowość, minimum i testy

Każdy portfel przeszedł automatyczny audyt: brak ujemnej wolnej gotówki i ujemnej ekspozycji, equity = cash + zablokowany koszt, jeden debit zakupu i jedna wypłata na trade, cash debit = gross + fee cash, brutto−netto = fee shares, payout zgodny z wynikiem, brak przyszłego receive/source eventu przy T−59 i zgodność dziennych log-growth z końcowym kapitałem. Opłata w udziałach zmniejsza payout; fee cash zwiększa debit; oba nie są naliczane drugi raz.

Testy regresyjne: `python -m unittest discover -s tests -p 'test_btc_preopen_t59_reassessment.py'` — 22 testy, OK (21.1s). Obejmują rzeczywiste wznowienie po częściowym checkpointcie, zgodność serial/parallel, historyczny filtr i idempotencję metadanych pokrycia.

Minimalny rozmiar pochodzi z bieżącego pola Gamma `orderMinSize` jeżeli jest dostępne; fallback to 5 udziałów. Brak historycznego archiwum minimów oznacza, że to przybliżenie wykonawcze, nie pomiar historyczny. Nie obniżamy zlecenia przy braku głębokości i nie zastępujemy wybranej strony inną.

Czas: weryfikacja zapisanych wejść i PMXT 9.9s; odczyt indeksów i selektywna ekstrakcja archiwum 43.7s; kontynuacja replayu od checkpointu po 3,420 partycjach 9927.5s; portfele i diagnostyki 162.1s. Końcowe uruchomienie raportujące z ponownym użyciem zweryfikowanego artefaktu trwało 275.9s. Łącznego wall-clock czasu wcześniejszych prób i przerw nie zapisano.

## Zmiany względem historycznego wyniku

| Obszar | Przed | Po | Powód |
|---|---|---|---|
| Reported 4,830 rejection cohort | 4,830 full snapshots rejected; 2,283 recovered in the prior diagnosis | 2,283/4,830 currently priceable at $5; implementation signal 0; ask-side qualification 2,283 | The same cohort was re-evaluated against the verified replay; the old cache still lacks per-file content hashes and replay-code identity. |
| Separate historical T−59 full-invalid cohort | 5,817 full snapshots with quote_valid=false | 2,439 have a current $5 native-ask price | This historical snapshot file is a different cohort from the reported 4,830; old replay provenance is unverified. |
| Replay cache provenance | status=complete; reused=True; fingerprints_match=False | full replay of 50,184 indexed markets; content/schema, index, token map, configuration, logic and source identity verified | Old snapshot SHA proved integrity only; old manifest had no content dependency records or implementation identity. |
| T−59 snapshot population | 33,482 cached snapshots through 2026-08-10T00:00:00+00:00 | 50,184 snapshots through cutoff 2026-10-07T00:00:00+00:00 | Added causal Gamma markets and free public AG6/V3 history, with published archive gaps explicitly represented. |
| Reported +$3,872.40 | cap-$20 control PnL $3,872.40, unverified | cap-$20 control PnL $5,953.50, rebuilt | New data range and verified source lineage require a new portfolio path; the historical figure is retained but not certified. |
| Unlimited 5% free-cash policy | PnL $-74.20; ending cash $25.80 | PnL $-74.20; ending cash $25.80 | Recomputed from verified snapshots and continuous settlement accounting; July controls quantify sizing and selection effects. |

## Odtworzenie i pliki

Raport i małe tabele są wersjonowane w `reports/btc_preopen/t59_reassessment_20261008`. Duże dane pozostają lokalnie w `data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008`; ich rozmiary i SHA256 są w `data_manifest.json`. Replay dependency manifest zawiera hash każdej partycji. Wznawianie archiwum używa checkpointu `pmxt_extended/archive_checksums/public_archive_checkpoint.json`; uruchomienie `python run_btc_preopen_t59_reassessment.py` używa zapisanych Gamma/predykcji/Binance, wznawia brakujące publiczne partycje, a następnie odbudowuje zależny replay i portfele.

Trading: wyłączony. Fit modelu i nowe przeszukiwanie polityk: nie uruchamiano. Wyniki pozostają retrospektywne i nie wybierają ani nie aktywują polityki.

Źródła: [PMXT V2](https://archive.pendulumflow.com/pmxt/v2/?page=1), [AG6](https://archive.pendulumflow.com/third-party/ag6/), [AG6 format/licencja](https://archive.pendulumflow.com/formats/ag6), [V3](https://archive.pendulumflow.com/v3/).
