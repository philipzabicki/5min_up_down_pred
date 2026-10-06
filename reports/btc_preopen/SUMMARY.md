# BTC pre-open: ocena eksperymentu i telemetria live

Główny scenariusz ekonomiczny to ustalone wejście T−59 s: sekundę po nominalnej decyzji T−60 s. To założenie operacyjne dla przyszłego uruchomienia serwera. Nie jest zmierzonym maksimum ani gwarantowanym worst case. Warianty wcześniejszych analiz pozostają w `economic_scenarios.csv`; dalsze porównania w tym raporcie dotyczą T−59 s.

Nie uruchomiono ponownego pobrania archiwum ani pełnego replayu. Ocenę księgi i porównanie ekonomiczne zbudowano z istniejącego `entry_snapshots.parquet`, `data_coverage.csv`, `trades.parquet` i zapisanych podsumowań/checkpointów. Nie wysłano prawdziwych zleceń, nie aktywowano kandydata ani handlu live.

## Czas i pochodzenie ceny wejścia

Tak: archiwum mapuje każde `condition_id` do natywnych tokenów UP/DOWN przez indeks rynku i zapisane mapowanie tokenów. Oficjalny start T pochodzi z indeksu rynku / bucketa sluga; dla przykładowych rynków poniżej slug epoch zgadza się z T. Replay używa aktualizacji według `timestamp_received` kolektora archiwum. Snapshot T−59 obejmuje zdarzenia odebrane do tej chwili włącznie, w tym zmiany rozmiaru i usunięcia poziomów; później odebrane zdarzenia są wykluczone nawet wtedy, gdy ich czas źródłowy wygląda na wcześniejszy. Zdarzenia z czasem źródłowym po wejściu są także odrzucane przez kontrolę przyczynowości. PMXT nie podaje monotonicznego identyfikatora kolejności: zdarzenia z identycznymi receive/source timestamp mają tylko stabilny porządek w części Parquet, nie gwarantowaną kolejność giełdową. Odtworzony best ask porównano z raportowanym BBO; niezgodne snapshoty są wykluczane.

Pełny `book` jest inicjalizatorem stanu, nie ceną zakupu. Po nim replay składa stan z wcześniejszych zmian poziomów. Fill $5 przechodzi po natywnych poziomach ask właściwego tokena, uwzględniając dostępną głębokość i opłaty; komplementowane kwotowanie nie dostarcza głębokości do fillu. Zapisany replay raportuje 0 snapshotów skażonych zdarzeniami odebranymi po wejściu i 0 snapshotów ze zdarzeniem źródłowym po wejściu.

Przy T−60 oba natywne booki były zainicjalizowane dla 9,399/9,407 rynków; dla 2,422 BBO obu stron były poprawne, a dla 9,383 obie strony miały głębokość wystarczającą na $5. Przy T−59, filtrze ask age 30 s i pozostałych warunkach kwalifikuje się 2,331/9,407 rynków; powody z cache: `{"incomplete_or_crossed_book": 7000, "eligible": 2331, "stale_ask": 29, "unreconciled_best_ask_at_entry": 26, "exchange_maintenance_pause": 12, "no_initial_book_snapshot": 8, "unknown_historical_fee": 1}`.

`ask age` to czas od ostatniej zmiany poziomu po stronie ask w natywnej księdze: dodanie, zmiana rozmiaru/ceny albo usunięcie poziomu odświeża wiek, także gdy zmienił się poziom poza best ask. Aktualizacja rozmiaru przy tej samej cenie odświeża go tylko, jeśli rozmiar faktycznie się zmienił; identyczny duplikat nie. Usunięcie best ask odświeża wiek i przesuwa BBO na następny poziom. Pełny snapshot resetuje wiek; gdy źródłowy timestamp jest niedostępny, kod używa czasu odbioru. Osobne `ask_received_age` liczy od ostatniej zmienionej głębokości po czasie odbioru archiwizatora. Filtr 30 s ogranicza wiek zmienionej głębokości ask według czasu źródłowego; nie mierzy opóźnienia wejścia i sam nie dowodzi, że lokalny feed nie miał przerwy.

Przerwa między ostatnią wiadomością odebraną przez archiwizator a wejściem T−60 miała p50 0.726 s, p95 3.027 s, p99 4.688 s i maksimum 1460.334 s. To cisza w archiwalnym odbiorze, nie dowód braku zdarzeń na giełdzie ani jakość feedu hipotetycznego serwera. Pierwsza obserwacja archiwalna oznacza pierwsze zdarzenie zobaczone przez eksportera, nie moment publikacji rynku przez giełdę.

Próbki początku, środka i końca okresu: ceny w kolumnie to best bid/best ask, a kwota po średniku to zrekonstruowany VWAP zakupu $5. Czasy ask pokazują odbiór kolektora i czas źródłowy ostatniej zmiany głębokości.

| Okres | Rynek / condition_id | T | Wejście | Pierwsza obserwacja archiwalna | Inicjalizacja booka UP / DOWN | Ostatnie zdarzenie odebrane przed wejściem | Ostatnia zmiana ask: UP / DOWN (odbiór; źródło; wiek) | Natywne BBO, rozmiar ask i cena $5 |
|---|---|---|---|---|---|---|---|---|
| start | `0x8ae0371ab2021a553ee3f7a237ee701964bf2e017ae8a6c0d78fb43179db95d5` / `btc-updown-5m-1776273000` | 2026-04-15T17:10:00+00:00 | 2026-04-15T17:09:01+00:00 | 2026-04-14T17:18:40.315000+00:00 | 2026-04-14T17:18:40.315000+00:00 / 2026-04-14T17:18:40.315000+00:00 | 2026-04-15T17:09:00.213000+00:00 | 2026-04-15T17:09:00.213000+00:00 (2026-04-15T17:09:00.152000+00:00; age 0.848s) / 2026-04-15T17:09:00.213000+00:00 (2026-04-15T17:09:00.143000+00:00; age 0.857s) | UP `370033696666235685419298806385131140639346262331292027808378867135689806352`: 0.50/0.51, 237.92 shares, $0.51; DOWN `67103309535592287189501612156910505345968746590311581144200559228030100874285`: 0.49/0.50, 66.28 shares, $0.50 |
| middle | `0x1d878aafde224a0a51b8b19b659ba25cfda278091502ac142d28a95c6de8a020` / `btc-updown-5m-1777709400` | 2026-05-02T08:10:00+00:00 | 2026-05-02T08:09:01+00:00 | 2026-05-01T08:19:57.271000+00:00 | 2026-05-01T08:19:57.519000+00:00 / 2026-05-01T08:19:57.271000+00:00 | 2026-05-02T08:09:00.803000+00:00 | 2026-05-02T08:09:00.398000+00:00 (2026-05-02T08:09:00.181000+00:00; age 0.819s) / 2026-05-02T08:09:00.803000+00:00 (2026-05-02T08:09:00.769000+00:00; age 0.231s) | UP `10948058240696486467760596317525385801555556257940310085544727184016277564112`: 0.50/0.51, 327.11 shares, $0.51; DOWN `89177715602471616393884692807780562978133582114179351081076537736623080492159`: 0.49/0.50, 416.66 shares, $0.50 |
| end | `0xc425822cd9e6d800c58a6f4d746c3c2c599dda905702d1bbfc0ff24f919d6053` / `btc-updown-5m-1779098700` | 2026-05-18T10:05:00+00:00 | 2026-05-18T10:04:01+00:00 | 2026-05-17T10:16:48.953000+00:00 | 2026-05-17T10:16:48.953000+00:00 / 2026-05-17T10:16:48.953000+00:00 | 2026-05-18T10:04:00.773000+00:00 | 2026-05-18T10:04:00.773000+00:00 (2026-05-18T10:04:00.574000+00:00; age 0.426s) / 2026-05-18T10:04:00.773000+00:00 (2026-05-18T10:04:00.585000+00:00; age 0.415s) | UP `16318614289463896926700534259463742166195403303017234099910627446003340294313`: 0.50/0.51, 227.24 shares, $0.51; DOWN `88468605374990090251663312259504715883703699633896263522862389849874996522962`: 0.49/0.50, 873.27 shares, $0.50 |

Te archiwalne `timestamp_received` pochodzą od eksportera, nie od naszego przyszłego serwera. Live używa obecnie REST `/book` dla obu tokenów; znacznik odbioru jest lokalny po pobraniu i parsowaniu odpowiedzi, a źródłowy timestamp zostaje pusty, jeśli API go nie zwraca. Kod nie utrzymuje jeszcze strumienia booka. Nie utożsamiam tych zegarów.

## Porównanie ekonomiczne T−59 s

Każdy wiersz stosuje te same dostępne snapshoty, filtr ask age ≤30 s, zakup brutto $5, początkową gotówkę $100, model historycznej opłaty i zwrot kapitału 60 s po rozstrzygnięciu. Strategie wybierają transakcje niezależnie; wspólny zbiór oznacza te same kwalifikujące się rynki, a nie wymuszone identyczne transakcje.

`MARKET_ONLY` nie ma zapisanego, porównywalnego portfela pre-open. Z cache policzono dwa nietrenowane baseline’y bez informacji BTC: stałe p=0.5 i p=0.5015364895 (prewalencja development). Nie stroiłem ich do okresu. Candidate_platt daje na wspólnych rynkach PnL $318.16 wobec -$95.11 dla oryginalnego Platt; baseline’y dają odpowiednio -$52.09 i -$97.54. To dodatni wynik tego replayu, nie potwierdzenie niezależnej przewagi.

| Zakres | Model | PnL netto | Kapitał końcowy | Drawdown | Transakcje | Obrót | Opłaty | Odrzucenia danych / bez przewagi / brak salda |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all archived (9,407) | `candidate_platt` | $318.16 | $418.16 | 40.67% | 1,076 | $5380.00 | $0.50 | 7,076 / 1,255 / 0 |
| all archived (9,407) | `original_v1_platt` | -$95.11 | $4.89 | 96.75% | 287 | $1435.00 | $0.00 | 7,076 / 1,208 / 836 |
| all archived (9,407) | `candidate_raw` | $355.53 | $455.53 | 31.42% | 1,059 | $5295.00 | $0.50 | 7,076 / 1,272 / 0 |
| all archived (9,407) | `original_v1_raw` | -$99.43 | $0.57 | 99.61% | 373 | $1865.00 | $28.32 | 7,076 / 1,035 / 923 |
| all archived (9,407) | `no_btc_constant_0_5` | -$52.09 | $47.91 | 92.97% | 119 | $595.00 | $0.00 | 7,076 / 2,212 / 0 |
| all archived (9,407) | `no_btc_development_prevalence_0_5015364895` | -$97.54 | $2.46 | 97.55% | 52 | $260.00 | $0.00 | 7,076 / 2,173 / 106 |
| common eligible (2,331) | `candidate_platt` | $318.16 | $418.16 | 40.67% | 1,076 | $5380.00 | $0.50 | 0 / 1,255 / 0 |
| common eligible (2,331) | `original_v1_platt` | -$95.11 | $4.89 | 96.75% | 287 | $1435.00 | $0.00 | 0 / 1,208 / 836 |
| common eligible (2,331) | `candidate_raw` | $355.53 | $455.53 | 31.42% | 1,059 | $5295.00 | $0.50 | 0 / 1,272 / 0 |
| common eligible (2,331) | `original_v1_raw` | -$99.43 | $0.57 | 99.61% | 373 | $1865.00 | $28.32 | 0 / 1,035 / 923 |
| common eligible (2,331) | `no_btc_constant_0_5` | -$52.09 | $47.91 | 92.97% | 119 | $595.00 | $0.00 | 0 / 2,212 / 0 |
| common eligible (2,331) | `no_btc_development_prevalence_0_5015364895` | -$97.54 | $2.46 | 97.55% | 52 | $260.00 | $0.00 | 0 / 2,173 / 106 |

Księgowanie odtworzono z kodu symulatora i zapisanych wpisów T−59: gotówka przed wejściem musi pokryć pełny debet (`$5 + fee w collateral`, bez kredytu), kapitał jest blokowany do `resolved_at_utc + 60 s`, a po ostatnim rynku symulator rozlicza wszystkie pozostałe pozycje. W 2,795 zapisanych transakcjach znaleziono 0 ujemnych stanów gotówki i 0 naruszeń pokrycia debetu; minimum po wejściu wyniosło $0.57. Legacy fee zmniejsza liczbę udziałów; współczesna opłata jest debetowana w collateral. Dokładne zaokrąglenie maker-level nie jest dostępne w zagregowanym booku. Drawdown liczy gotówkę plus koszt zablokowanych pozycji, bez mark-to-market.

Candidate_platt przewyższa oryginalny model Platt i oba proste baseline’y w zapisanej symulacji. Nie istnieje jednak porównywalny wyuczony baseline `MARKET_ONLY` w tej samej pre-open definicji; wcześniejsze wyniki MARKET_ONLY mają inny moment decyzji/feature availability i nie są podstawiane do tej tabeli.

## Chronologia i niezależność oceny

Fit i dobór iteracji kończyły się przed 2026-01-01T00:00:00+00:00; ostatnia dostępna etykieta treningu to 2025-12-31T23:59:00+00:00. Kalibracja Platt używała etykiet dostępnych do 2026-04-15 17:00 UTC, a pierwsza decyzja okresu testowego była 17:04 UTC (T rynku 17:05). Kandydat wybierano na wcześniejszych foldach Q3 2025. Nie wykryto bezpośredniego przecieku etykiety do predykcji na badanych punktach czasu.

Mimo tej chronologii okres ekonomiczny 2026-04-15–2026-05-18 był już użyty w wcześniejszych eksperymentach repozytorium: mieści się w foldach selekcji/tuningu innych komponentów i był raportowany jako development. Wynik ekonomiczny jest więc retrospektywnym wynikiem rozwojowym, a nie niezależnym, nietkniętym holdoutem. Przyczynowość rekonstrukcji booka i niezależność wyboru modelu to odrębne własności.

Tę tabelę policzono bez dostrajania strategii do okresu. Przyjęto istniejącą regułę dodatniego oczekiwanego zwrotu netto, age cap 30 s, gross $5 i zwolnienie kapitału po 60 s. Nie ma w repozytorium zamrożonego, datowanego przed okresem protokołu potwierdzającego niezależny wybór tych ekonomicznych ustawień; z uwagi na historyczne użycie danych nie traktuję ich wyniku jako prospektywnego testu strategii.

## Telemetria opóźnień live

Każda decyzja zachowuje istniejący rekord CSV oraz identyfikator decision/condition/token/model; dołączone są czasy UTC danych Binance (source i receive), gotowości cech, predykcji, decyzji i końca cyklu. Księga zapisuje początek requestu, źródłowy timestamp jeśli istnieje, lokalny odbiór, pochodzenie, stan synchronizacji REST oraz BBO UP/DOWN. Submit zapisuje start/koniec wywołania, osobny lokalny odbiór odpowiedzi, order ID i status/powód pominięcia. Czas lokalnych etapów nadal mierzy monotoniczny `perf_counter`; timestampy zdarzeń są UTC.

Po cyklu log `[latency_summary]` podaje N, p50/p95/p99, maksimum, liczbę przekroczeń budżetu 1 s i ujemnych różnic dla każdego obserwowalnego etapu względem nominalnego T−60; liczniki rozdzielają cykle, próby, odpowiedzi klienta, order IDs, pola fill zgłoszone w odpowiedzi oraz niezależne rekordy zdarzeń fill. Nie wolno odczytywać mediany wszystkich cykli jako opóźnienia prób zlecenia ani sumować percentyli etapów.

Send po warstwie transportowej, źródłowy ACK giełdy oraz źródłowy i lokalnie odebrany fill pozostają puste: obecny synchroniczny CLOB client nie udostępnia tu tych zdarzeń, a user stream fill nie jest podłączony. HTTP/client response i dodatni `filled_stake_usdc` nie są czasem giełdowego ACK ani dowodem niezależnie timestampowanego fillu. Endpoint CLOB `/time` zapisuje jedynie przybliżony offset względem czasu hosta; RTT, NTP status i niepewność offsetu nie są mierzone.

W dotychczasowych, innych runtime’ach: 527 cykli miało close-to-cycle p50/p95/p99 475/1271/1792 ms; 111 synchronicznych submitów miało p50/p95/p99 403/631/1002 ms. Dwie kolekcje pre-open miały wszystkie wejścia gotowe 2553 i 3661 ms po decyzji i miały wyłączone zlecenia. To odrębne historyczne pomiary, nie podstawa do wyboru T−59 i nie pomiary kandydata end-to-end.

Instrukcja lokalizacji kolumn, odczytu `[latency_summary]` i rozróżnienia czasu send/ACK/fill jest w [`docs/live_telemetry.md`](../../docs/live_telemetry.md).

## Gotowość kandydata do live

Nie. Bundle kandydata wymaga 112 cech, pre-open collector ma obecnie bundle 29 cech, a ogólny runtime 256 kolumn, z których tylko 44 pokrywają się z kandydatem. Zmierzony warm inference dotyczy już zbudowanego wektora i nie obejmuje aktualizacji cech. Zgodna inkrementalna ścieżka obliczania 112 cech pozostaje osobnym brakiem wdrożeniowym; kandydata nie aktywowano.

## Artefakty

- `primary_economic_comparison.csv` — T−59, wszystkie rynki i wspólny zbiór kwalifikujących się rynków.
- `book_timing_examples.csv` — audyt trzech ksiąg T−59 z archiwum lokalnego.
- `economic_scenarios.csv` — zachowana wcześniejsza macierz wariantów czasowych i freshness.
- `audit.json`, `data_coverage.csv`, `trades.parquet`, `model_comparison.csv` i `report_bundle.zip` — szczegóły oraz materiały odtwarzalności.

Nie złożono rzeczywistych zleceń ani nie zmieniono konfiguracji aktywnego modelu.
