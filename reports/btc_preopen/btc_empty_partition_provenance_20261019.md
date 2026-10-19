# Pochodzenie pustych partycji PMXT BTC 5m — pilot 2026-10-19

## Wniosek

W trzech wskazanych, obecnie dostępnych oryginalnych archiwach PMXT jest łącznie 137 377 951 wierszy, ale **nie ma ani jednego dokładnego dopasowania `market` do żadnego z 33 482 rynków w zamrożonym indeksie**. Dotyczy to godzin 2026-06-13T12, 2026-06-20T12 i 2026-06-28T12. Dopasowanie wynosi zero już przed warunkami czasu; oryginalny selektor i niezależny filtr referencyjny zgadzają się. W tych trzech obecnych obiektach nie znaleziono zdarzeń rynków, które miałyby zostać pominięte przez lokalną ekstrakcję. Nie potwierdził się błąd selekcji.

Ustalenie obejmuje wyłącznie te trzy próbki. **Nie dowodzi**, że pozostałe godziny luki też nie zawierają zdarzeń. Historyczny checkpoint nie zapisał hashy, ETagów ani wersji oryginałów, dlatego nie da się stwierdzić, czy zdalna zawartość zmieniła się od ekstrakcji. Stary pusty wynik pozostaje nieweryfikowalny co do tożsamości źródła.

Nie pobrano pozostałych 381 plików. Nie odzyskano zdarzeń; predykcje, wykonania, rejestry decyzji i PnL pozostały niezmienione.

## Zamrożona próbka i zakres selekcji

Konfigurację zapisano przed pierwszym żądaniem sieciowym: [`pmxt_empty_partition_pilot_20261019_config.json`](pmxt_empty_partition_pilot_20261019_config.json), SHA-256 `2a866cd064488bde532efac3e10adcbc92f2b7b5070641d413ea3122696df85a`. Godziny wybrano jako początek, środek i koniec luki, bez odwołania do wyników transakcji.

Oryginalny ekstraktor czytał pliki godzinowe z zakresem historii 25 godzin, dopasowywał `market` do całego indeksu 33 482 rynków i stosował cutoff `timestamp_received <= market_start_utc - 59 s` włącznie. Nie ograniczyłem kontroli do rynków rozpoczynających się w godzinie pliku. Dla kontroli okna 25 h konfiguracja zawiera odpowiednio 311, 310 i 311 startów rynku oraz ich cutoffy. Wszystkie grupy wierszy były kandydatami według statystyk `market` dla pełnego indeksu, więc do selekcji odczytano każdą grupę.

## Wynik odczytu oryginałów

| Godzina UTC | URL | Rozmiar obiektu | Wiersze wg metadanych / odczytane | Grupy odczytane | Dopasowania `market` przed czasem | Po cutoffie odbioru | Po dolnej granicy 25 h | Po cutoffie `timestamp` | Wybrane zdarzenia |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-06-13T12 | [PMXT](https://r2v2.pmxt.dev/polymarket_orderbook_2026-06-13T12.parquet) | 312 186 409 B | 54 758 083 / 54 758 083 | 53/53 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-20T12 | [PMXT](https://r2v2.pmxt.dev/polymarket_orderbook_2026-06-20T12.parquet) | 257 972 776 B | 46 127 510 / 46 127 510 | 44/44 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-28T12 | [PMXT](https://r2v2.pmxt.dev/polymarket_orderbook_2026-06-28T12.parquet) | 205 991 888 B | 36 492 358 / 36 492 358 | 35/35 | 0 | 0 | 0 | 0 | 0 |

Pierwsza kolumna czasu to oryginalny cutoff ekstraktora: `timestamp_received <= market_start_utc - 59 s`, włącznie. Kontrole po dolnej granicy 25 h i po `timestamp` są dodatkowymi testami referencyjnymi; nie zostały przedstawione jako dodatkowe warunki historycznego selektora.

Dla wszystkich 132 grup dostępne były statystyki min/max `market`; wszystkie grupy zostały uznane za kandydatów pełnego indeksu i faktycznie odczytane. Brak dopasowań w części grup nie jest tu używany jako dowód o całym pliku: sprawdzono wszystkie grupy każdego z trzech obiektów.

Per plik zapisano URL, czas odczytu, rozmiar, ETag, Last-Modified, metadane Parquet, liczniki filtrów, SHA-256 lokalnej pustej partycji audytowej i przykładowe rekordy źródłowe. ETagi i czasy pobrania:

| Godzina | Pobrano UTC | ETag | Last-Modified |
|---|---|---|---|
| 2026-06-13T12 | 2026-10-19T20:48:48.763174+00:00 | `"a100f383136bbe64df4de7bd271581df-38"` | Tue, 23 Jun 2026 17:39:29 GMT |
| 2026-06-20T12 | 2026-10-19T20:51:51.201824+00:00 | `"ca4be1f89bbb688c6e65e017fccab6b3-31"` | Tue, 23 Jun 2026 22:46:43 GMT |
| 2026-06-28T12 | 2026-10-19T20:52:34.805614+00:00 | `"f17c232dd23e81f81ca8c1c6f9675fc0-25"` | Sun, 28 Jun 2026 13:06:37 GMT |

Są to ETagi obiektów multipart, nie deklaruję ich jako hashy. Pełnego SHA-256 zdalnych obiektów nie obliczono, bo pobierano zakresy bajtów, a nie całe obiekty. Sumaryczny rozmiar trzech obiektów to 776 151 073 B. Oszacowanie skompresowanych kolumn potrzebnych ekstraktorowi wyniosło 775 216 434 B.

## Zgodność identyfikatorów i czasów

Indeks zawiera `condition_id` jako 66-bajtowy tekst ASCII: małe litery, prefiks `0x` i 64 znaki hex. Źródło ma `market` typu `fixed_size_binary[66]`. Sprawdziłem wszystkie 137 377 951 wartości: każda miała długość 66, prefiks `0x`, małe litery i poprawne ASCII. Nie znaleziono dopasowania także przy niezależnym porównaniu bez rozróżniania wielkości liter ani po pominięciu prefiksu. Przykładowe rekordy dla każdej godziny, z identyfikatorami `market` i `asset_id`, zachowano w manifeście audytu.

`timestamp_received` i `timestamp` mają typ `timestamp[ms, tz=UTC]`. W każdym z 137 377 951 wierszy `timestamp` był nie późniejszy od `timestamp_received`. Największe zaobserwowane opóźnienie `timestamp` względem czasu odbioru wynosiło 5 365 ms dla 13 czerwca, 2 137 503 258 ms dla 20 czerwca i 2 828 537 697 ms dla 28 czerwca. To wskazuje, że `timestamp_received` jest osią czasu odbioru/partycjonowania, a `timestamp` może opisywać znacznie starszy czas źródłowy; cutoff ekstraktora prawidłowo opiera się na `timestamp_received`. Ponieważ nie było dopasowanych rynków, nie da się osobno zweryfikować tych czasów dla zdarzeń naszych rynków.

Schemat źródła ma `asset_id` i `event_type` typu string. Liczba wybranych wierszy, poprawnych `asset_id` i rozpoznanych typów zdarzeń wynosi zero, bo nie wybrano żadnego zdarzenia. Nie jest to test mapowania tokenów UP/DOWN dla tych rynków.

## Porównanie selektorów i granice dowodu

Oryginalna selekcja Arrow (`market` exact membership, potem cutoff w milisekundach) i niezależny filtr referencyjny używający surowych wartości bajtowych oraz bezpośrednich dat UTC zwróciły identyczne pozycje wierszy; różnic było zero. Liczniki kolejnych filtrów czasu są zerowe dlatego, że przed nimi nie było dopasowania rynku, a nie dlatego, że cutoff odrzucił znalezione zdarzenia.

Obecna zawartość trzech plików wyjaśnia puste partycje tych godzin: oryginały zawierają zdarzenia innych rynków, ale nie zawierają identyfikatorów z zamrożonego indeksu. Nie ma dowodu na błąd filtra ani na zmianę zdalnego obiektu od czasu starego checkpointu. Dla pozostałych 381 godzin wynik pozostaje nieznany.

Historyczny `hour_checkpoint.json` nadal potwierdza wyłącznie, że zapisany ekstraktor nie zachował zdarzeń. Nie wiąże pustych partycji z wersją lub hashem źródła. Starego cache nie usuwano ani automatycznie nie pobierano. W nowym manifeście jego stan oznaczono jako `unverifiable_source_identity_and_filter_counters`.

## Naprawione raportowanie dostępności

`fixed5_skip_reason` rozróżnia teraz istniejący snapshot T−59 bez zdarzeń jako `no_archive_events_observed_by_t59`. Ten warunek ma pierwszeństwo przed ogólnym powodem braku godziny archiwum. W zapisanym `coverage_calendar.parquet` nowa etykieta występuje przy 8 285 pustych snapshotach. W kalendarzu BASE70 dokładnie 4 602 wiersze z 13–28 czerwca, które wcześniej opisywał raport, mają teraz tę etykietę zamiast `archive_hour_missing_at_entry`; naprawiono też kopie diagnostyczne w kalendarzach BASE70 information-value i raw holdout. Rejestrów decyzji ani wykonania nie zmieniano.

W dziennych tabelach development dodano jawne `market_count`, `available_input_count`, liczbę decyzji i wykonań per wariant oraz `data_status`. `available_input_count` to liczba rynków kwalifikujących się w zapisanym builderze cech; decyzje liczą wszystkie wiersze ledgeru, także skipy; wykonania liczą wiersze ledgeru fill.

W 47 dniach ekonomicznych było 0 dni bez rynku, 16 dni z rynkami, ale bez dostępnych wejść, 4 dni z decyzjami i bez wykonań oraz 27 dni z wykonaniami RAW. C, X i MARKET zachowują po 5 083 decyzje i 0 wykonań; RAW zachowuje 782 wykonania. Zera PnL C/X/MARKET oznaczają więc decyzje bez transakcji w dniach z dostępnymi wejściami albo brak dostępnych wejść w 16 dniach luki. Zapisane wartości PnL porównano przed i po regeneracji i zachowano dokładnie.

## Koszt sprawdzenia pozostałych godzin

Trzy obiekty miały 206–312 MB; średnia daje około **98,6 GB** pełnych bajtów dla pozostałych 381 plików. Pomiar skompresowanych kolumn ekstraktora daje około **98,5 GB**. Przy zmierzonym czasie odczytu i filtrowania (średnio 56,7 s na próbkę, sekwencyjnie) prostoliniowa ekstrapolacja to około **6 godzin**. To przybliżenie z trzech celowo wybranych godzin, nie gwarancja rozmiaru ani czasu dla całej luki.

Łączny konserwatywny górny limit bajtów pilota wyniósł 1 395 477 253 B (1,30 GiB) z dozwolonych 2 GiB. Licznik obejmuje powtórne zakresy, metadane, retry po HTTP 520 i rezerwę 128 MiB dla przerwanej pierwszej próby lokalnego filtra referencyjnego. Nie przekroczono limitu. Nie uruchamiano odbudowy pozostałych 381 godzin.

## Artefakty i dalsza odbudowa

- Konfiguracja przed odczytem: [`pmxt_empty_partition_pilot_20261019_config.json`](pmxt_empty_partition_pilot_20261019_config.json).
- Pełny [manifest pilota](btc_empty_partition_pilot_20261019/pilot_manifest.json) zawiera przykładowe rekordy i liczniki; [preflight](btc_empty_partition_pilot_20261019/preflight.json) zachowuje schemat i statystyki wszystkich grup wierszy. Pełna tabela 384 godzin z uzupełnionymi danymi pilota jest w [CSV godzinowym](btc_empty_partition_provenance_20261019_hourly.csv). Rejestry HTTP Range i przerwanych prób pozostają lokalnie w `data/analysis/polymarket/BTC/btc_market_microstructure_development_20261019/availability/pmxt_empty_partition_pilot_20261019/`.
- Nowe partycje audytowe `recovered_events/{hour}.parquet` mają po 0 wierszy i nie nadpisują historycznych partycji. Nie było zdarzeń do zapisania jako odzyskane artefakty.
- Zaktualizowano diagnostykę `coverage_calendar.parquet`, trzy kalendarze BASE70 oraz `availability/daily_funnel.csv`, `economics/daily_pnl.csv` i raport development. Predykcje, ledger decyzji/wykonań i PnL nie wymagają odbudowy.
- Pozostaje nieweryfikowalne, czy brak zdarzeń w trzech aktualnych obiektach opisuje też historyczne bajty użyte przez dawny ekstraktor. Rozstrzygnięcie pozostałych 381 godzin wymaga osobnego ograniczonego odczytu; obecny pilot nie uzasadnia pobrania ich automatycznie.

Nie uruchamiano treningu, strojenia, eksperymentu C/X, pełnego replayu ani masowej odbudowy.
