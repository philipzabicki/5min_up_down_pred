# Stan eksperymentu BTC entry-state — 2026-10-04

## Rozstrzygnięcie

Nie da się uczciwie ocenić C ani D na danych historycznych dostępnych lokalnie. W 9 407 wspólnych oknach nie ma ceny poziomu startowego z feedu rozstrzygającego Polymarket ani jej czasu dostępności. Nie dopasowałem więc C/D, nie policzyłem dla nich przedziałów bootstrapu i nie dopisałem im wyników portfela.

Istniejące porównanie A/B wskazuje małą, development-only poprawę po dodaniu nowego modelu BTC do rynku: Δ log loss B−A = −0,000235 (95% CI [−0,000464, −0,000064]), Δ Brier = −0,000121 (95% CI [−0,000231, −0,000035]). Przedziały pochodzą ze sparowanego moving-block bootstrapu, bloki 3-dniowe, 2 000 replikacji. Okres nie jest niezależnym testem: brał udział w doborze iteracji nowego modelu, selekcji cech i dopasowaniu profili.

Na wcześniej zamrożonych wejściach BTC-only prawdopodobieństwo wybranej strony miało średnią 50,60%, a trafność wyniosła 39,26% (318/810). Na tych samych 810 wejściach A uzyskał log loss 0,659506 i Brier 0,233481; B — 0,659594 i 0,233503. A i B są tu tylko diagnozą na próbie wybranej przez wcześniejszą politykę, nie niezależnym porównaniem strategii. Aktualny stan BTC nie został oceniony.

Wniosek operacyjny: pozostawić A jako benchmark prognozy i B jako porównawczy wariant z poprzedniego badania; nie promować nowej polityki wejść. Kontynuować wyłącznie zbieranie danych z nową sesją bez transakcji. C/D pozostają nierozstrzygnięte do czasu uzyskania zsynchronizowanych, punktowych danych z właściwego feedu.

## Oś czasu i definicje

| Zdarzenie | Definicja użyta w istniejącym replayu | Pokrycie / uwaga |
| --- | --- | --- |
| Początek rynku | `market_start_utc`; granica 5-minutowego rynku BTC | 9 407/9 407, zakres 2026-04-15 17:05–2026-05-18 10:30 UTC |
| Koniec rynku | `market_end_utc = market_start_utc + 5 min` | 9 407/9 407; `seconds_to_expiry` przy decyzji = 300 dla wszystkich |
| Rozstrzygnięcie | Oficjalny wynik Polymarket; payload Gamma podaje `resolutionSource=https://data.chain.link/streams/btc-usd-twap-60s-streams` | 9 407/9 407 wyników. To Chainlink BTC/USD TWAP 60 s, nie cena Binance. Wynik BTC proxy różni się w 405/9 407 okien (4,31%). |
| Koniec wejścia wcześniejszego BTC | Dokładny UTC `Opened` w minutowym OOF | Każdy `Opened = market_start − 1 min`; to zamknięta świeca Binance COIN-M BTCUSD index, której koniec przypada na start rynku. |
| Dostępność OOF | `Opened + 1 min = decision_available_at = market_start` | Zgodne dla 9 407 wierszy. To założenie historycznego replayu; rzeczywisty czas policzenia nowej predykcji w shadow jest późniejszy. |
| Kwotowanie Polymarket | Pierwszy prawidłowy snapshot Kacho w chwili decyzji lub do 2 s po dostępności OOF | 9 407/9 407; zapisany `quote_delay_ms` ma 0 ms. Wiek giełdowego booka Kacho jest nieznany. |
| Zamrożona decyzja | `decision_available_at`, równe początkowi rynku | Nie przesuwałem decyzji w poszukiwaniu lepszego wyniku. |
| Symulowane wykonanie | Ask z pierwszego zapisanego kwotowania po 1 s, zgodnie z istniejącym replayem | To założenie fillu, nie rzeczywisty fill. Koszty, głębokość i ograniczenie gotówki są wspólne dla wariantów. |
| Settlement i gotówka | Wynik jest osobnym zdarzeniem; portfel uwalnia kapitał dopiero po późniejszym rozstrzygnięciu | Nie utożsamiam `market_end` z dostępnością wypłaty. |

W live shadow oba feedy docierają po granicy rynku. W zachowanej sesji v2 mediana faktycznego odbioru świecy BTC wynosiła 9,36 s po starcie (zakres 1,25–16,77 s), predykcji 9,60 s, a kwotowania 9,75 s. Ten pomiar nie zmienia historycznej definicji OOF, ale oznacza, że prospektywnego modelu ani późnego kwotowania nie wolno przypisywać do decyzji w chwili startu.

## Pokrycie danych historycznych

| Składnik | Okna z danymi | Źródło / dostępność | Ocena |
| --- | ---: | --- | --- |
| Oficjalny target i czas rozstrzygnięcia | 9 407 | Zapisane rynki Polymarket; rozstrzygnięcie z Gamma | Kompletne |
| Stary i nowy BTC OOF, identyfikator modelu/foldu | 9 407 | Zapisany wspólny Parquet; modele `20261002_041540` i `20261003_043549` | Kompletne dla A/B; okres development |
| Kwotowania, bid/ask, rozmiary i opłaty | 9 407 | Kacho; snapshot w chwili dostępności OOF; opłaty zapisane w dataset | Kompletne dla historycznego replayu; rzeczywisty wiek booka nieznany |
| Cena bieżąca BTC | 9 407 możliwych świec | Binance COIN-M BTCUSD index 1m; zamknięcie świecy `Opened=market_start−1m` | Proxy wobec Chainlink; w historycznym pliku brak faktycznego czasu odbioru tej świecy |
| Poziom odniesienia rynku w chwili decyzji | **0/9 407** | Nie ma go w schemacie wspólnego datasetu ani w zachowanym payloadzie Gamma | Brak cechy dla C/D |
| Dokładny historyczny feed Chainlink TWAP 60 s | **0 okien z potwierdzonym feedem i czasem dostępności** | `resolutionSource` wskazuje TWAP 60 s, lecz lokalne archiwa dotyczą innego feedu lub nie mają point-in-time timestampów | Niewystarczające dla cechy poziomu i standaryzowanej odległości |
| Czas do końca | 9 407 | Różnica czasu końca i decyzji | Stałe 300 s; nie wnosi zmienności |
| Zmienność historyczna | 9 407 możliwych wartości proxy | 30 wcześniejszych log-zwrotów 1m z Binance index; odchylenie próby × `sqrt(seconds_left/60)` | Causalna proxy. Nie uzupełnia brakującego poziomu odniesienia. |

Sprawdziłem istniejące archiwa oraz ścieżki historycznego pobierania w `data/chainlink_sources.py`:

- `BTCUSD_reports.csv`: 1 426 raportów od 2026-03-21 18:51:45 do 21:26:57 UTC; brak nakładania na wspólny zbiór.
- Publiczne archiwum minutowe z 2026-04-02: 1 438 wierszy z 2026-04-01 12:54 do 2026-04-02 12:51 UTC; brak nakładania.
- Publiczne archiwum minutowe z 2026-05-08: 1 438 wierszy z 2026-05-07 13:58 do 2026-05-08 13:55 UTC; pokrywa 288 początków rynku. To stream `BTC/USD-RefPrice-DS-Premium-Global-003`, a nie wskazany przez Gamma stream TWAP 60 s. Plik zawiera opóźnione OHLC, bez czasu dostępności każdej obserwacji.
- Publiczny endpoint użyty przez projekt przyjmuje `feedId`, `abiIndex` i `timeRange`; dla historii minutowej wybiera ostatni bucket `1D`. Obsługa punktowego historycznego API Candlestick ma parametry `from`/`to`, ale wymaga poświadczeń `CHAINLINK_CANDLESTICK_USER_ID` i `CHAINLINK_CANDLESTICK_API_KEY`; w tym środowisku nie są skonfigurowane. Nie próbowałem obchodzić uwierzytelnienia ani używać innego dostawcy.

Nie używałem `Close`, `High` ani `Low` minuty zawierającej decyzję do odtwarzania ceny po jej zakończeniu. Wspólne dane mają przed decyzją zamkniętą świecę Binance z poprzedniej minuty, ale nie mają ceny ani czasu dostępności poziomu startowego Chainlink.

## A/B na wspólnych oknach

| Wariant | N | Log loss | Brier | Kalibracja slope / intercept |
| --- | ---: | ---: | ---: | ---: |
| A — MARKET_ONLY, L2 logistic | 9 407 | 0,688384 | 0,247547 | 0,923 / −0,012 |
| B — MARKET + BTC_MODEL | 9 407 | 0,688148 | 0,247426 | 0,946 / −0,011 |
| C — MARKET + CURRENT_STATE | 0 z kompletnym stanem | — | — | Nie dopasowano |
| D — MARKET + CURRENT_STATE + BTC_MODEL | 0 z kompletnym stanem | — | — | Nie dopasowano |

Chronologiczne log loss A/B:

| Okres | N | MARKET_ONLY | MARKET + BTC_MODEL |
| --- | ---: | ---: | ---: |
| F0, 2026-04-15 17:05–2026-04-26 15:50 UTC | 3 136 | 0,688066 | 0,688107 |
| F1, 2026-04-26 15:55–2026-05-07 13:10 UTC | 3 135 | 0,691973 | 0,691741 |
| F2, 2026-05-07 13:15–2026-05-18 10:30 UTC | 3 136 | 0,685113 | 0,684598 |

Przydział cech, preprocessing, regularyzacja i walidacja A/B pochodzą z istniejącego, zamrożonego porównania. Nie stroiłem nowych progów ani LightGBM. C/A, D/C i D/B są **nieestymowalne**, a nie równe zeru. Na bieżących danych nie da się też uczciwie dopasować wspólnego czterowariantowego walk-forward ani sparowanego bootstrapu tych różnic.

## Zamrożone wejścia wybrane przez wcześniejszą politykę BTC

Istniejący audyt `entry_selection_audit_20261003` zamroził 810 sygnałów strategii `new_btc_platt`; wyników na tej próbce nie używam jako niezależnego testu.

| Prawdopodobieństwo dla wybranej strony | Średnia p | Trafność / log loss / Brier |
| --- | ---: | ---: |
| BTC-only `new_btc_platt` (polityka wybierająca próbkę) | 0,50604 | Trafność 318/810 = 0,39259; LL 0,693412; Brier 0,250132 |
| A — MARKET_ONLY | 0,40511 | LL 0,659506; Brier 0,233481 |
| B — MARKET + new BTC | 0,41003 | LL 0,659594; Brier 0,233503 |
| Surowy market midpoint | 0,41280 | LL 0,660514; Brier 0,233969 |

Na tej samej próbce A nieznacznie wygrywa z B (Δ LL B−A = +0,000088; Δ Brier = +0,000023). Średnie BTC-only p dla zakupionej strony przewyższa faktyczną trafność o 11,35 pp. To warunkowa kalibracja na wejściach wybranych przez tę politykę; bez C/D nie wiadomo, czy stan BTC to koryguje.

Niezależny nominalny replay tych 810 wejść miał obrót 4 050 USD, opłaty 166,34 USD i PnL po kosztach −383,72 USD (−9,47% obrotu). Przy jednym portfelu 100 USD wykonano 73 sygnały, 737 odrzucono z braku gotówki, zapłacono 14,70 USD opłat, a końcowy kapitał wyniósł 2,11 USD. Są to hipotetyczne fill’e z istniejącego replayu, nie rzeczywiste transakcje.

## Istniejący replay ekonomiczny A/B

Zamrożona polityka: początkowe 100 USD, stawka 5 USD, bufor EV po opłatach 0,25 USD, ten sam wybór kwotowań i opłat, 1 s do symulowanego wykonania, ograniczenie do widocznej płynności, gotówka blokowana do oficjalnego settlementu. W wynikach `market_plus_new_btc` wykonał 5 transakcji vs 4 dla `market_only`; dodatkowy zakup UP 2026-04-28 04:35 UTC po 0,33 przegrał i odpowiada za −5 USD różnicy wyniku.

| Wariant | Transakcje | Obrót | Opłaty | PnL / obrót | Kapitał końcowy | Maks. DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MARKET_ONLY | 4 | 20 USD | 0,63 USD | 26,2% | 105,24 USD | 4,54% |
| MARKET + new BTC | 5 | 25 USD | 0,87 USD | 1,0% | 100,24 USD | 6,44% |

W tym zbiorze wynik nie dowodzi dodatniej wartości po kosztach modelu BTC. Replay nie ma rzeczywistych fill’i ani rzeczywistego wieku feedu Kacho. Ekonomiki C/D nie oceniłem, bo nie powstały prawidłowe predykcje tych wariantów.

## ADX: pięć znanych różnic

Przyczyną była adaptacyjna końcowa średnia MAMA liczona po DX. Offline liczy ją od początku dostępnej historii, a wcześniejszy runtime resetował ją na końcu okna `IndicatorWindowScratch`; zwiększenie tego okna do 100 000 nie usuwało różnicy.

| UTC | Offline / pełna historia | Poprzedni runtime | Δ prawdopodobieństwa UP |
| --- | ---: | ---: | ---: |
| 2026-09-04 16:04 | 14,960815 | 13,981462 | 0,000403938 |
| 2026-09-08 14:39 | 72,339351 | 71,450363 | 0,000203450 |
| 2026-09-04 15:59 | 24,188266 | 21,484080 | 0,000176763 |
| 2026-09-13 12:04 | 56,302434 | 52,676906 | 0,000043259 |
| 2026-09-16 09:14 | 79,073166 | 70,780350 | 0,000017451 |

Ponowne obliczenie tych pięciu wierszy pełną historią odtwarza zapisane wartości offline. Nie zmienił się żaden sygnał UP/DOWN. Nie można potwierdzić przejścia progu EV ani decyzji biznesowej: dla tych timestampów audyt nie ma historycznych kwotowań. Nie zwiększałem tolerancji. Nowy helper `features/live_indicator_runtime_entry_state_v1.py` używa dokładnie batchowego `get_adx_values` na pełnym, dostępnym prefiksie historii; osobny collector nadpisuje tylko tę cechę. Stary runtime v2 pozostaje niezmieniony.

Test celowany: `python -m unittest tests/test_entry_state_adx_runtime.py`. Test potwierdza, że reset MAMA na krótszym oknie różni się od batchu, a helper pełnej historii daje identyczną wartość.

## Collector i stan projektu

- Dotychczasowa sesja `btc_new_model_shadow_v2_20261004` jest zachowana bez zmian. Jej baza zawiera 56 rynków / 168 rekordów wariantów od 2026-10-03 23:15 do 2026-10-04 03:50 UTC; proces nie działał podczas audytu.
- Sesja `btc_entry_state_collection_v2_20261004` ma osobny protokół, manifest i SQLite: `data/analysis/polymarket/BTC/entry_state_collection_20261004_v2/`. Działa w trybie `collection_only`, bez zleceń i redeemów.
- Do 2026-10-04 04:55 UTC zapisała dwa okna (04:50 i 04:55), oba bez transakcji. W obu znacznik czasu CLOB wyprzedzał lokalny czas odbioru o ok. 0,19–0,21 s; reguła świeżości oznaczyła kwotowanie jako nieważne. Nie poszerzałem limitu ani nie traktowałem takich kwotowań jako wykonalnych.
- Każdy nowy rekord zachowuje świecę proxy BTC, faktyczny czas odbioru i predykcji, źródło/URL rozstrzygnięcia, czas i wiek booka, pełne kwotowanie, opłaty, 30-minutową zmienność i 300 s do końca. Brak poziomu startowego Chainlink zapisuje jako `null` wraz z powodem.
- Prawdopodobieństwa MARKET_ONLY i MARKET+BTC wyliczane z późnego kwotowania przechowywane są w osobnym polu `market_model_probabilities_at_late_quote` z flagą niekwalifikującą ich do zamrożonej decyzji startowej. Główne pola tych wariantów pozostają `null`; collector nie symuluje wejść.
- Początkowa próba uruchomienia collection v1 zakończyła się przed pierwszą decyzją przez błąd jednostki czasu pandas; powstałe świeczki i manifest są w odrębnym `entry_state_collection_20261004_v1`, nie są użyte w analizie. Poprawka konwersji i nowa sesja v2 mają osobny hash.

Manifest eksperymentu: `docs/btc_entry_state_experiment_manifest_20261004.json`. Manifest runtime v2 zapisuje hashe protokołu, kodu, modelu i danych.

## Odtworzenie

- Wcześniejsze A/B z zapisanych artefaktów: `python run_btc_new_model_comparison.py`.
- Test ADX: `python -m unittest tests/test_entry_state_adx_runtime.py`.
- Uruchomienie nowego collection-only: `python run_btc_entry_state_collection.py` (bez argumentów CLI; konfiguracja w `configs/btc_entry_state_collection_protocol_20261004_v2.json`).
