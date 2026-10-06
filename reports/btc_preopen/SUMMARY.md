# BTC pre-open: ocena eksperymentu i telemetria live

Główny scenariusz ekonomiczny to ustalone wejście T−59 s: sekundę po nominalnej decyzji T−60 s. To założenie operacyjne dla przyszłego uruchomienia serwera. Nie jest zmierzonym maksimum ani gwarantowanym worst case. Warianty wcześniejszych analiz pozostają w `economic_scenarios.csv`; dalsze porównania w tym raporcie dotyczą T−59 s.

Nie pobierano ponownie archiwum. Ukierunkowany replay lokalnych partycji PMXT odtworzył historię dla kompletnych, semantycznie poprawnych booków T−60/T−59, aby zweryfikować świeżość referencji BBO; zakres i zasoby są zapisane w raporcie. Ekonomikę policzono ponownie z istniejących snapshotów i filli, bo korekta zmienia kwalifikację rynków. Nie wysłano prawdziwych zleceń i nie uruchomiono handlu live.

Weryfikacja BBO objęła 4,591 rynków / 9,157 snapshotów, odczytała 810 lokalnych partycji, zastosowała 26,984,593 zdarzeń do stanów docelowych i trwała 118.7 min. Szczyt RSS próbkowany co 100 ms wyniósł 2086 MiB; nie było ruchu sieciowego ani pobierania danych.

## Czas i pochodzenie ceny wejścia

Tak: archiwum mapuje każde `condition_id` do natywnych tokenów UP/DOWN przez indeks rynku i zapisane mapowanie tokenów. Oficjalny start T pochodzi z indeksu rynku / bucketa sluga; dla przykładowych rynków poniżej slug epoch zgadza się z T. Replay używa aktualizacji według `timestamp_received` kolektora archiwum. Snapshot T−59 obejmuje zdarzenia odebrane do tej chwili włącznie, w tym zmiany rozmiaru i usunięcia poziomów; później odebrane zdarzenia są wykluczone nawet wtedy, gdy ich czas źródłowy wygląda na wcześniejszy. Zdarzenia z czasem źródłowym po wejściu są także odrzucane przez kontrolę przyczynowości. PMXT nie podaje monotonicznego identyfikatora kolejności: zdarzenia z identycznymi receive/source timestamp mają tylko stabilny porządek w części Parquet, nie gwarantowaną kolejność giełdową. Referencję BBO uznajemy za rozstrzygającą tylko, gdy jej czas źródłowy jest późniejszy od ostatniej zmiany obu stron. Starsza lub równa referencja jest raportowana osobno, bo bez identyfikatora sekwencji nie ustala kolejności; świeża rozbieżność ask wyklucza snapshot.

Pełny `book` jest inicjalizatorem stanu, nie ceną zakupu. Po nim replay składa stan z wcześniejszych zmian poziomów. Fill $5 przechodzi po natywnych poziomach ask właściwego tokena, uwzględniając dostępną głębokość i opłaty; komplementowane kwotowanie nie dostarcza głębokości do fillu. Zapisany replay raportuje 0 snapshotów skażonych zdarzeniami odebranymi po wejściu i 0 snapshotów ze zdarzeniem źródłowym po wejściu.

Przy T−60 oba natywne booki były zainicjalizowane dla 9,399/9,407 rynków; dla 4,588 BBO obu stron były poprawne, a dla 9,383 obie strony miały głębokość wystarczającą na $5. Przy T−59, filtrze ask age 30 s i pozostałych warunkach kwalifikuje się 4,454/9,407 rynków; powody z cache: `{"incomplete_or_crossed_book": 4830, "eligible": 4454, "unreconciled_best_ask_at_entry": 71, "stale_ask": 31, "exchange_maintenance_pause": 12, "no_initial_book_snapshot": 8, "unknown_historical_fee": 1}`.
The former 75% rejection rate was 75.2% (7,076/9,407), not proof that those markets had no exchange liquidity. The earlier strict `bid < ask` rule rejected valid locked books; after accepting locks, 2,370 qualified (+39 vs the old strict stage). The old-BBO stage then rejected 2,157 ask mismatches. The timestamp gate counted 4,190 prior BBO disagreements cleared as older/tied across T-60 and T-59; at T-59 only 71 fresh ask mismatches remain, and 4,454 qualify (+2,084 net). Current exclusions are 4,953, including 4,830 incomplete/crossed books. The full primary-reason sums and overlapping flags are in the audit tables; none of these counts establish exchange liquidity where the archive is incomplete.

`ask age` to wiek ostatniej rzeczywistej zmiany dowolnego poziomu ask w natywnej księdze: dodanie, zmiana ceny/rozmiaru albo usunięcie poziomu odświeża go, także poza best ask; identyczny duplikat nie. Pełny snapshot inicjalizuje oba booki i resetuje wiek. Zmiana rozmiaru ≤0 usuwa poziom. Osobne `ask_received_age` używa czasu odbioru archiwizatora, a filtr 30 s korzysta z czasu źródłowego (zastępowanego czasem odbioru, jeśli źródła brak). To wiek zmienionej głębokości, nie opóźnienie wejścia ani miara ciągłości feedu.

Przerwa między ostatnią wiadomością odebraną przez archiwizator a wejściem T−60 miała p50 0.726 s, p95 3.027 s, p99 4.688 s i maksimum 1460.334 s. To cisza w archiwalnym odbiorze, nie dowód braku zdarzeń na giełdzie ani jakość feedu hipotetycznego serwera. Pierwsza obserwacja archiwalna oznacza pierwsze zdarzenie zobaczone przez eksportera, nie moment publikacji rynku przez giełdę.

Kontrola schematu wykazała 811 lokalnych partycji PMXT i 1 wariantów kolumn; pole `schema_version` i monotoniczny event sequence ID nie występują. Liczniki archiwum to `{"book": 617342, "price_change": 102390694, "last_trade_price": 1429897}` (globalnie, nie tylko dla odrzuconych rynków). Dla T−59 porównano referencję BBO w 4,569 kwalifikowanych do tej kontroli wpisach: 3 starszych i 8,795 równych czasowo aktualizacji strony. Rozkład przyczyn odrzuceń i przykłady w `book_rejection_markets.csv`/`BOOK_TRACE_EXAMPLES.md` wskazują na stan inicjalizacji, crossed book, świeżość ask i jakość uzgodnienia BBO; nie ma podstaw, by przypisać je do wariantu schematu.

Próbki początku, środka i końca okresu: ceny w kolumnie to best bid/best ask, a kwota po średniku to zrekonstruowany VWAP zakupu $5. Czasy ask pokazują odbiór kolektora i czas źródłowy ostatniej zmiany głębokości.

| Okres | Rynek / condition_id | T | Wejście | Pierwsza obserwacja archiwalna | Inicjalizacja booka UP / DOWN | Ostatnie zdarzenie odebrane przed wejściem | Ostatnia zmiana ask: UP / DOWN (odbiór; źródło; wiek) | Natywne BBO, rozmiar ask i cena $5 |
|---|---|---|---|---|---|---|---|---|
| start | `0x8ae0371ab2021a553ee3f7a237ee701964bf2e017ae8a6c0d78fb43179db95d5` / `btc-updown-5m-1776273000` | 2026-04-15T17:10:00+00:00 | 2026-04-15T17:09:01+00:00 | 2026-04-14T17:18:40.315000+00:00 | 2026-04-14T17:18:40.315000+00:00 / 2026-04-14T17:18:40.315000+00:00 | 2026-04-15T17:09:00.213000+00:00 | 2026-04-15T17:09:00.213000+00:00 (2026-04-15T17:09:00.152000+00:00; age 0.848s) / 2026-04-15T17:09:00.213000+00:00 (2026-04-15T17:09:00.143000+00:00; age 0.857s) | UP `370033696666235685419298806385131140639346262331292027808378867135689806352`: 0.50/0.51, 237.92 shares, $0.51; DOWN `67103309535592287189501612156910505345968746590311581144200559228030100874285`: 0.49/0.50, 66.28 shares, $0.50 |
| middle | `0x1d878aafde224a0a51b8b19b659ba25cfda278091502ac142d28a95c6de8a020` / `btc-updown-5m-1777709400` | 2026-05-02T08:10:00+00:00 | 2026-05-02T08:09:01+00:00 | 2026-05-01T08:19:57.271000+00:00 | 2026-05-01T08:19:57.519000+00:00 / 2026-05-01T08:19:57.271000+00:00 | 2026-05-02T08:09:00.803000+00:00 | 2026-05-02T08:09:00.398000+00:00 (2026-05-02T08:09:00.181000+00:00; age 0.819s) / 2026-05-02T08:09:00.803000+00:00 (2026-05-02T08:09:00.769000+00:00; age 0.231s) | UP `10948058240696486467760596317525385801555556257940310085544727184016277564112`: 0.50/0.51, 327.11 shares, $0.51; DOWN `89177715602471616393884692807780562978133582114179351081076537736623080492159`: 0.49/0.50, 416.66 shares, $0.50 |
| end | `0xc425822cd9e6d800c58a6f4d746c3c2c599dda905702d1bbfc0ff24f919d6053` / `btc-updown-5m-1779098700` | 2026-05-18T10:05:00+00:00 | 2026-05-18T10:04:01+00:00 | 2026-05-17T10:16:48.953000+00:00 | 2026-05-17T10:16:48.953000+00:00 / 2026-05-17T10:16:48.953000+00:00 | 2026-05-18T10:04:00.773000+00:00 | 2026-05-18T10:04:00.773000+00:00 (2026-05-18T10:04:00.574000+00:00; age 0.426s) / 2026-05-18T10:04:00.773000+00:00 (2026-05-18T10:04:00.585000+00:00; age 0.415s) | UP `16318614289463896926700534259463742166195403303017234099910627446003340294313`: 0.50/0.51, 227.24 shares, $0.51; DOWN `88468605374990090251663312259504715883703699633896263522862389849874996522962`: 0.49/0.50, 873.27 shares, $0.50 |

Te archiwalne `timestamp_received` pochodzą od eksportera, nie od naszego przyszłego serwera. Live używa obecnie REST `/book` dla obu tokenów; znacznik odbioru jest lokalny po pobraniu i parsowaniu odpowiedzi, a źródłowy timestamp zostaje pusty, jeśli API go nie zwraca. Kod nie utrzymuje jeszcze strumienia booka. Nie utożsamiam tych zegarów.

## Porównanie ekonomiczne T−59 s

Każdy wiersz stosuje te same dostępne snapshoty, filtr ask age ≤30 s, zakup brutto $5, początkową gotówkę $100, model historycznej opłaty i zwrot kapitału 60 s po rozstrzygnięciu. Strategie wybierają transakcje niezależnie; wspólny zbiór oznacza te same kwalifikujące się rynki, a nie wymuszone identyczne transakcje.

`MARKET_ONLY` nie ma zapisanego, porównywalnego portfela pre-open. Z cache policzono dwa nietrenowane baseline’y bez informacji BTC: stałe p=0.5 i p=0.5015364895 (prewalencja development). Nie stroiłem ich do okresu. Candidate_platt daje na wspólnych rynkach PnL $475.84 wobec -$95.11 dla oryginalnego Platt; baseline’y dają odpowiednio -$97.99 i -$95.71. To dodatni wynik tego replayu, nie potwierdzenie niezależnej przewagi.

| Zakres | Model | PnL netto | Kapitał końcowy | Drawdown | Transakcje | Obrót | Opłaty | Odrzucenia danych / bez przewagi / brak salda |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all archived (9,407) | `candidate_platt` | $475.84 | $575.84 | 66.27% | 2,316 | $11580.00 | $1.00 | 4,953 / 2,138 / 0 |
| all archived (9,407) | `original_v1_platt` | -$95.11 | $4.89 | 96.71% | 765 | $3825.00 | $1.99 | 4,953 / 2,075 / 1,614 |
| all archived (9,407) | `candidate_raw` | $384.19 | $484.19 | 74.88% | 2,297 | $11485.00 | $0.50 | 4,953 / 2,157 / 0 |
| all archived (9,407) | `original_v1_raw` | -$97.70 | $2.30 | 97.94% | 206 | $1030.00 | $61.94 | 4,953 / 1,829 / 2,419 |
| all archived (9,407) | `no_btc_constant_0_5` | -$97.99 | $2.01 | 98.73% | 349 | $1745.00 | $0.00 | 4,953 / 3,553 / 552 |
| all archived (9,407) | `no_btc_development_prevalence_0_5015364895` | -$95.71 | $4.29 | 98.20% | 606 | $3030.00 | $0.00 | 4,953 / 3,053 / 795 |
| common eligible (4,454) | `candidate_platt` | $475.84 | $575.84 | 66.27% | 2,316 | $11580.00 | $1.00 | 0 / 2,138 / 0 |
| common eligible (4,454) | `original_v1_platt` | -$95.11 | $4.89 | 96.71% | 765 | $3825.00 | $1.99 | 0 / 2,075 / 1,614 |
| common eligible (4,454) | `candidate_raw` | $384.19 | $484.19 | 74.88% | 2,297 | $11485.00 | $0.50 | 0 / 2,157 / 0 |
| common eligible (4,454) | `original_v1_raw` | -$97.70 | $2.30 | 97.94% | 206 | $1030.00 | $61.94 | 0 / 1,829 / 2,419 |
| common eligible (4,454) | `no_btc_constant_0_5` | -$97.99 | $2.01 | 98.73% | 349 | $1745.00 | $0.00 | 0 / 3,553 / 552 |
| common eligible (4,454) | `no_btc_development_prevalence_0_5015364895` | -$95.71 | $4.29 | 98.20% | 606 | $3030.00 | $0.00 | 0 / 3,053 / 795 |

## Wpływ walidacji kwotowań na ekonomikę T−59

Tabela porównuje trzy etapy filtracji przy tych samych zapisanych cenach, fillach, opłatach i zasadach gotówki. Pierwszy odtwarza dawną walidację bid < ask; drugi dopuszcza poprawne bid = ask przy starym uzgadnianiu BBO; trzeci stosuje korektę świeżości referencji. Każdy wiersz pokazuje rynki kwalifikowane w danym etapie. Pełna tabela sześciu modeli/baseline’ów dla wszystkich rynków, zbiorów kwalifikowanych i ich przecięcia oraz lista zmian per rynek są w CSV.

Dopuszczenie zablokowanych kwotowań dodało 39 kwalifikowane rynki przed korektą referencji BBO. Korekta świeżości dodała 2,084 i odrzuciła 0 po potwierdzeniu świeżej rozbieżności. Na wspólnym zbiorze wynik każdego z sześciu modeli jest identyczny we wszystkich trzech etapach.
Poprzednie +$318.16 pozostaje wynikiem na 2,331 rynkach wspólnych dla trzech walidacji; skorygowany zbiór obejmuje 4,454 kwalifikowanych rynków i daje $475.84 dla candidate_platt. Różnica wynika ze zmienionej kwalifikacji snapshotów, nie ze zmiany modelu ani strategii.

| Walidacja | Rynki kwalifikowane | Model | PnL netto | Transakcje |
|---|---:|---|---:|---:|
| `original_strict_bid_lt_ask` (2,331) | `candidate_platt` | $318.16 | 1,076 |
| `original_strict_bid_lt_ask` (2,331) | `original_v1_platt` | -$95.11 | 287 |
| `original_strict_bid_lt_ask` (2,331) | `candidate_raw` | $355.53 | 1,059 |
| `original_strict_bid_lt_ask` (2,331) | `original_v1_raw` | -$99.43 | 373 |
| `original_strict_bid_lt_ask` (2,331) | `no_btc_constant_0_5` | -$52.09 | 119 |
| `original_strict_bid_lt_ask` (2,331) | `no_btc_development_prevalence_0_5015364895` | -$97.54 | 52 |
| `locked_quotes_old_bbo_freshness` (2,370) | `candidate_platt` | $323.01 | 1,099 |
| `locked_quotes_old_bbo_freshness` (2,370) | `original_v1_platt` | -$95.11 | 287 |
| `locked_quotes_old_bbo_freshness` (2,370) | `candidate_raw` | $355.39 | 1,083 |
| `locked_quotes_old_bbo_freshness` (2,370) | `original_v1_raw` | -$99.23 | 371 |
| `locked_quotes_old_bbo_freshness` (2,370) | `no_btc_constant_0_5` | -$97.54 | 36 |
| `locked_quotes_old_bbo_freshness` (2,370) | `no_btc_development_prevalence_0_5015364895` | -$97.54 | 54 |
| `locked_quotes_corrected_bbo_freshness` (4,454) | `candidate_platt` | $475.84 | 2,316 |
| `locked_quotes_corrected_bbo_freshness` (4,454) | `original_v1_platt` | -$95.11 | 765 |
| `locked_quotes_corrected_bbo_freshness` (4,454) | `candidate_raw` | $384.19 | 2,297 |
| `locked_quotes_corrected_bbo_freshness` (4,454) | `original_v1_raw` | -$97.70 | 206 |
| `locked_quotes_corrected_bbo_freshness` (4,454) | `no_btc_constant_0_5` | -$97.99 | 349 |
| `locked_quotes_corrected_bbo_freshness` (4,454) | `no_btc_development_prevalence_0_5015364895` | -$95.71 | 606 |

Księgowanie odtworzono z poprawionych wpisów T−59: gotówka przed wejściem musi pokryć pełny debet (`$5 + fee w collateral`, bez kredytu), kapitał jest blokowany do `resolved_at_utc + 60 s`, a po ostatnim rynku symulator rozlicza wszystkie pozostałe pozycje. W 2,316 transakcjach `candidate_platt` znaleziono 0 ujemnych stanów gotówki i 0 naruszeń pokrycia debetu; minimum po wejściu wyniosło $30.28. Legacy fee zmniejsza liczbę udziałów; współczesna opłata jest debetowana w collateral. Dokładne zaokrąglenie maker-level nie jest dostępne w zagregowanym booku. Drawdown liczy gotówkę plus koszt zablokowanych pozycji, bez mark-to-market.

Candidate_platt przewyższa oryginalny model Platt i oba proste baseline’y w zapisanej symulacji. Nie istnieje jednak porównywalny wyuczony baseline `MARKET_ONLY` w tej samej pre-open definicji; wcześniejsze wyniki MARKET_ONLY mają inny moment decyzji/feature availability i nie są podstawiane do tej tabeli.

## Chronologia i niezależność oceny

Fit i dobór iteracji kończyły się przed 2026-01-01T00:00:00+00:00; ostatnia dostępna etykieta treningu to 2025-12-31T23:59:00+00:00. Kalibracja Platt używała etykiet dostępnych do 2026-04-15 17:00 UTC, a pierwsza decyzja okresu testowego była 17:04 UTC (T rynku 17:05). Kandydat wybierano na wcześniejszych foldach Q3 2025. Nie wykryto bezpośredniego przecieku etykiety do predykcji na badanych punktach czasu.

Mimo tej chronologii okres ekonomiczny 2026-04-15–2026-05-18 był już użyty w wcześniejszych eksperymentach repozytorium: mieści się w foldach selekcji/tuningu innych komponentów i był raportowany jako development. Wynik ekonomiczny jest więc retrospektywnym wynikiem rozwojowym, a nie niezależnym, nietkniętym holdoutem. Przyczynowość rekonstrukcji booka i niezależność wyboru modelu to odrębne własności.

Tę tabelę policzono bez dostrajania strategii do okresu. Przyjęto istniejącą regułę dodatniego oczekiwanego zwrotu netto, age cap 30 s, gross $5 i zwolnienie kapitału po 60 s. Nie ma w repozytorium zamrożonego, datowanego przed okresem protokołu potwierdzającego niezależny wybór tych ekonomicznych ustawień; z uwagi na historyczne użycie danych nie traktuję ich wyniku jako prospektywnego testu strategii.

## Telemetria opóźnień live

Każda decyzja zachowuje istniejący rekord CSV oraz identyfikator decision/condition/token/model; dołączone są czasy UTC danych Binance (source i receive), gotowości cech, predykcji, decyzji i końca cyklu. Księga zapisuje początek requestu, źródłowy timestamp jeśli istnieje, lokalny odbiór, pochodzenie, stan synchronizacji REST oraz BBO UP/DOWN. Submit zapisuje start/koniec wywołania, osobny lokalny odbiór odpowiedzi, order ID i status/powód pominięcia. Czas lokalnych etapów nadal mierzy monotoniczny `perf_counter`; timestampy zdarzeń są UTC.

Po cyklu log `[latency_summary]` podaje N, p50/p95/p99, maksimum, liczbę przekroczeń budżetu 1 s i ujemnych różnic dla każdego obserwowalnego etapu względem nominalnego T−60; liczniki rozdzielają cykle, próby, odpowiedzi klienta, order IDs, pola fill zgłoszone w odpowiedzi oraz niezależne rekordy zdarzeń fill. Nie wolno odczytywać mediany wszystkich cykli jako opóźnienia prób zlecenia ani sumować percentyli etapów.

The authenticated Polymarket user stream records order placement updates and partial/final fills, with REST resync and order/attempt linking. It starts only for enabled live submit. Current official protocol docs and mocked reconnect tests were checked, but py-clob-client-v2 is not installed here, so there was no real handshake or observed exchange ACK/fill. A live order is needed to measure acceptance latency; only actual execution events can establish fill time, price, quantity, and any reported fee. HTTP/client response is not an exchange ACK or fill timestamp.

W dotychczasowych, innych runtime’ach: 527 cykli miało close-to-cycle p50/p95/p99 475/1271/1792 ms; 111 synchronicznych submitów miało p50/p95/p99 403/631/1002 ms. Dwie kolekcje pre-open miały wszystkie wejścia gotowe 2553 i 3661 ms po decyzji i miały wyłączone zlecenia. To odrębne historyczne pomiary, nie podstawa do wyboru T−59 i nie pomiary kandydata end-to-end.

Instrukcja lokalizacji kolumn, odczytu `[latency_summary]` i rozróżnienia czasu send/ACK/fill jest w [`docs/live_telemetry.md`](../../docs/live_telemetry.md).

## Gotowość kandydata do live

Kandydat ma osobny, nieaktywny paper runtime ze ścieżkami modelu, kalibratora, uporządkowanych 112 cech i konfiguracji historii. Nie zmieniono nazw, pozycji ani definicji cech: lista dokładnie zgadza się z oryginalnym v1, a zmieniły się wyuczony booster/kalibrator i parametry. Audyt mapuje rodziny {"candle": 47, "reaction_profile": 19, "volume_profile": 16, "realized_volatility": 9, "session": 7, "basis_premium": 6, "streak": 4, "indicator": 4} i sprawdza 85 historycznych decyzji względem rebuildów ograniczonych do chwili decyzji. Wektory miały 0 różnic cech, 0 różnic maski i 0 błędów wznowienia. Na lokalnym CPU p50/p95/p99 wyniosły: warm update 0.56/1.04/1.16 ms, pełny wektor 14.06/20.67/26.65 ms, predykcja z Platt 0.14/0.22/0.29 ms, cała ścieżka update→wektor→predykcja 14.85/22.74/28.00 ms. Szczyt RSS próbkowany co 100 ms: 2921 MiB. To potwierdza zgodność badanego lokalnego runtime, nie sprawdza bieżącego live feedu ani złożenia zlecenia. Kandydata nie aktywowano.

The exact 44/112 was a name intersection: the active BTC model bundle has 256 columns, of which 44 names occur in the candidate; the other 68 are absent from that separate bundle, not unsupported code. The 29-feature pre-open baseline is a separate causal raw-candle model (29 features; 1 exact name overlap). Candidate and original have the same ordered feature list (True), indicator-fit directory (True), and volume/reaction profile configs (True/True); no names, positions, or definitions changed.
Chaikin SHMMA accumulated numerical drift over more than 3 million candles. The fix serializes and incrementally advances its recurrence state; it does not reset a short window or relax tolerance. The candidate seed is valid through 2026-10-02T18:00:00+00:00; a later startup requires a complete contiguous closed-candle catch-up and fails before prediction if that interval is missing.
## Artefakty

- `primary_economic_comparison.csv` — T−59, wszystkie rynki i wspólny zbiór kwalifikujących się rynków.
- `book_timing_examples.csv` — audyt trzech ksiąg T−59 z archiwum lokalnego.
- `economic_scenarios.csv` — wcześniejsza macierz wariantów czasowych; nie przeliczono jej po korekcie BBO.
- `quote_validation_economic_impact.csv` i `quote_validation_market_changes.csv` — wpływ trzech etapów walidacji BBO na ekonomikę i kwalifikację per rynek.
- `quote_validation_corrected_trades.parquet` — transakcje z finalnej T−59 walidacji dla wszystkich sześciu modeli/baseline’ów.
- `fresh_bbo_replay_summary.json`, `quote_validation_before_after.csv` i `fresh_bbo_entry_results.csv` — zakres, zasoby i wyniki ukierunkowanego replayu świeżości BBO.
- `runtime_compatibility_audit.json`, `feature_compatibility_112.csv`, `feature_definition_comparison.json`, `artifact_manifest.json` and `archive_partition_hashes.csv` - candidate feature map, numerical parity and reproducibility fingerprints.
- `book_rejection_markets.csv`, `book_rejection_daily.csv` i `BOOK_TRACE_EXAMPLES.md` — powody odrzuceń i przykładowe ścieżki księgi.
- `audit.json`, `data_coverage.csv`, `quote_validation_corrected_trades.parquet`, `model_comparison.csv` i `report_bundle.zip` — szczegóły oraz odtwarzalność finalnej walidacji.

Nie aktywowano kandydata ani nie złożono rzeczywistych zleceń; dodane ścieżki konfiguracji dotyczą osobnego profilu paper.

## Etap optymalizacji polityki wejścia i sizingu — stan na 2026-10-07

**Status: strojenie nieuruchomione z powodu braku krzywej głębokości dla stawek zmiennych.** Zachowuję stały zakup $5 jako odtworzony punkt odniesienia, nie jako zatwierdzoną politykę docelową. Nie wybrano nowej polityki i nie ma podstaw, by twierdzić, że brak poprawy został przetestowany.

### Odtworzenie punktu odniesienia

Na 4,454 rynkach zakwalifikowanych po skorygowanym uzgadnianiu BBO reguła kupuje najwyżej jedną stronę raz na rynek, w T−59 s. Dla każdej strony liczy `EV_USD = p_strony × udziały_netto_po_fee − pełny_debet_gotówkowy` z zapisanego, dokładnego fillu $5. Kupuje stronę z większym dodatnim EV; przy remisie UP; gdy oba EV są niedodatnie — `no_trade`. `extra_buffer = 0`; EV jest w USD na zlecenie, nie na udział. Opłata historyczna jest odejmowana w udziałach przed 2026-04-28 11:00 UTC, a później — poza godziną przerwy giełdowej — w collateral. Nie dolicza się historycznego minimum zlecenia, którego danych nie ma.

Ponowny przebieg decyzji na wszystkich 9,407 rekordach cache odtwarza zapisane strony, EV i księgowanie: 4,454 zakwalifikowane, 2,138 `no_trade` z powodu braku dodatniego EV, 2,316 transakcji i 0 odrzuceń z braku gotówki. Końcowa księga daje **+$475.840155635 USD**: saldo $100 → $575.840156; obrót $11,580; fee $0.99999992; maksymalne obsunięcie kosztowe 66.2678%; maksymalny koszt pozycji otwartych $15 (3 pozycje); najniższa wolna gotówka po wejściu $30.2817. Największa stawka $5 stanowiła maksymalnie 12.41% bieżącego kapitału kosztowego, a maksymalny czas pod wodą wyniósł 12 dni 19 godz. 37 min. Średni dzienny logarytmiczny wzrost 5.149% obejmuje pełne 34 dni i jest opisem historycznej ścieżki baseline’u, nie wynikiem selekcji polityki.

Starsze **+$318.16** dotyczy 2,331 rynków przy poprzedniej, ścisłej walidacji `bid < ask`. Nie jest celem strojenia ani porównaniem równych zbiorów: skorygowana kwalifikacja zwiększyła próbę do 4,454 rynków.

### Cache, jakość danych i ograniczenie głębokości

[`policy_opportunities.parquet`](policy_opportunities.parquet) zachowuje wszystkie 9,407 rynków — również kwalifikujące się rynki bez dawnego zakupu — z prawdopodobieństwem zamrożonego `candidate_platt`, wejściem, askami i rozmiarami najlepszego poziomu, fee, powodem kwalifikacji, wynikiem, czasem dostępności wyniku oraz zwolnieniem kapitału. 4,454 rekordy kwalifikują się; 4,953 są zachowane z powodami wykluczenia. Przy 4,353 kwalifikujących się księgach brak rozstrzygającego potwierdzenia BBO; to oznaczenie pozostało jawne i nie są one opisane jako niezależnie potwierdzone.

Cache zawiera natywny **najlepszy ask z rozmiarem**, ale nie drabinkę dalszych poziomów. Dodatkowo przechowuje tylko dokładny, zagregowany fill brutto $5 na stronę. Obie najlepsze ceny pokrywają $5 w 4,030 z 4,454 kwalifikujących się rynków; 141 z 2,316 transakcji baseline’u rzeczywiście przechodzi przez poziomy gorsze od najlepszego asku. Dla stake’u pośredniego między poziomami nie można odzyskać dokładnej ceny z samego VWAP $5. Nie znamy też historycznego `orderMinSize`, więc nie da się zweryfikować wykonalności mniejszych zleceń bez przenoszenia współczesnego minimum wstecz. Nie ekstrapoluję fillu liniowo ani nie odtwarzam strumienia PMXT, bo oba działania są poza zakresem polecenia. Wobec tego nie uruchamiam sizingu wymagającego nieobecnej krzywej wykonania.

### Zamrożony projekt oceny i status wyboru

Przed jakimkolwiek strojeniem zapisano trzy rodziny nowych polityk i 25 skończonych konfiguracji (`policy_trials.csv`): stały stake $1/$2.50 (baseline $5 zachowany osobno), stały udział wolnej gotówki 1%/2.5%/5% z limitem $5 oraz ułamkowy Kelly 25%/50% z limitem $5 i krokiem $0.01. Minimalny oczekiwany zwrot po kosztach: 0%/2%/5%, liczony jako oczekiwany zysk netto USD podzielony przez pełny debet. Strona dla sizingu stałego i procentowego jest wybierana po najwyższym EV netto dla faktycznej stawki. Kelly maksymalizowałby oczekiwany logarytm wzrostu na wolnej gotówce, z fee i limitem dostępnej głębokości. `no_trade` ma wzrost zero. Stawki powyżej $5 są wyłączone, bo cache nie dokumentuje takiej egzekucji.

[`policy_study.json`](policy_study.json) zamraża podziały chronologiczne i tożsamość wejść. Dla 4,454 kwalifikujących się rynków pierwsze 20% to historia początkowa, a kolejne okna 20% służą naprzemiennie jako wcześniejsza walidacja wyboru i trzy zewnętrzne bloki. Bloki zewnętrzne zaczynają się 2026-04-28 19:34, 2026-05-05 14:44 i 2026-05-11 23:59 UTC; liczby etykiet dostępnych ściśle przed refitem zapisano per blok. Etykieta jest dostępna od `max(resolved_at, expiry)`. Wszystkie portfele miałyby wspólny start i przenosiły gotówkę oraz nierozliczone pozycje między blokami. Ten okres był już wcześniej użyty w rozwoju, więc ewentualny wynik byłby retrospektywną oceną walk-forward, nie nowym nietkniętym holdoutem.

Z powodu braku krzywej ask **0/25 prób wykonano**; nie ma oceny trial→raport ani wyników zewnętrznych bloków polityki. Nie można więc stwierdzić, czy poprawa wystąpiłaby tylko wewnątrz strojenia czy także w kolejnych blokach, ani wyliczyć kosztu poprawy w ryzyku. [`policy.json`](policy.json) zapisuje kompletną regułę baseline’u, fee, księgowanie, ograniczenia i fingerprint wejść; pozostaje nieaktywna. Bieżący runtime nie został zmieniony, a handlu, shadow ani live nie uruchomiono.

Powtarzalność kontroli wejść: `python audit_btc_preopen_policy_readiness.py`. Skrypt czyta istniejące cache’e, porównuje predykcje z zapisanym plikiem `candidate_platt`, sprawdza sumę +$475.84, drawdown, brak podwójnych zakupów i zgodność czasu zwalniania kapitału. Model SHA-256: `19be11d61db42e32aa94c294b516d5ca615cde7377375f5eb265e57423943986`; pełne hashe plików wejściowych znajdują się w `policy_study.json`.
