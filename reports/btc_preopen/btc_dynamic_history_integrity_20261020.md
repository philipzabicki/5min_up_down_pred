# BTC dynamic-history integrity and availability — 2026-10-20

## Zakres i rozstrzygnięcie

Wykonano wyłącznie kontrolę integralności, dostępności danych i ochrony granic. Nie uruchamiano fitu, tuningu ani oceny ekonomicznej, nie pobierano pełnego archiwum, nie czytano wierszy holdoutu, nie zmieniano live i nie wysyłano zleceń. Zamrożony protokół `btc_market_dynamics_action_value_20261020` pozostał bez zmian.

**Etap jest zablokowany.** Lokalna historia nie wystarcza do spełnienia tego samego gate’u; wcześniejsze RAW OOF nie ma udokumentowanego czasu dostępności predykcji, a jego wcześniejsze wiersze zależą od selekcji cech i tuningu wykonanych na całym development. Nie znaleziono zweryfikowanego źródła, które dostarczałoby jednocześnie brakujące rynki, pełny historyczny strumień zmian księgi, wymagane snapshoty T−60/T−59 i causal RAW OOF. Nie przesunięto okresu ani progów.

## Zatwierdzone granice

Protokół w `data/analysis/polymarket/BTC/btc_market_dynamics_action_value_20261020/frozen_protocol.json` i jego konfiguracja ustalają:

- Rynek należy do development, gdy `market_start_utc < 2026-07-01T00:00:00Z`.
- Holdout zaczyna się włącznie `2026-07-01T00:00:00Z`; zamrożony zakres źródłowy kończy się wyłącznie `2026-10-07T00:00:00Z` (ostatni start holdoutu: `2026-10-06T23:55:00Z`). W tym audycie nie odczytano wierszy tego okresu.
- Ewaluacja development jest zamrożona na `[2026-05-15T00:00:00Z, 2026-07-01T00:00:00Z)`. Fold 1: `[2026-05-15, 2026-06-01)`, pierwsza decyzja `2026-05-14T23:59:00Z`. Fold 2: `[2026-06-01, 2026-07-01)`, pierwsza decyzja `2026-05-31T23:59:00Z`.
- Warm-up wynosi 30 dni, trening jest rozszerzającym się prefiksem, a etykiety treningowe muszą być oficjalnie opublikowane **ściśle przed** pierwszą decyzją T−60 danego foldu. `development_start_utc` jest zamrożoną regułą `auto_from_first_common_s_h_raw_market`; nie wyznaczono nowego startu.
- Pierwszy lokalnie udokumentowany wspólny start w użytym indeksie/statycznym zbiorze S+H+RAW to `2026-04-15T17:05:00Z`. To granica dostępności obecnych artefaktów wynikająca z manifestów, nie nowa granica eksperymentu.
- Zestaw H ma 21 niezmienionych cech: po siedem na horyzont 5 s, 15 s i 60 s — zmianę `p_market`, spreadów UP/DOWN, `log1p` notional asków UP/DOWN oraz względną deplecję asków UP/DOWN. Kolejność i progi gate’u pozostają zgodne z protokołem.

## Rozbieżność hashy cech

| Obiekt | SHA-256 / rozmiar | Pochodzenie i commit | Zawartość |
| --- | --- | --- | --- |
| Plik `features/btc_market_dynamics_action_value.py`, którego hash zapisano w zamrożonym protokole | `f7d6d74e241dbe3718e1a4cb0e9cc50407f7f4d5b829f7ebe124a4409c363d82`, 9 360 B | Zawartość odpowiada nieosiągalnemu blobowi Git `a015c3bbb2ccb4e5fbaef5a5c6a63c5f343031e3`. `git fsck` nie wykazał commitów zawierających ten blob; nie ma uczciwego commita źródłowego do przypisania temu zamrożonemu egzemplarzowi. | Kod źródłowy cech z jedną końcową dodatkową nową linią (pusty wiersz po ostatniej instrukcji). |
| Bieżący plik `features/btc_market_dynamics_action_value.py` i wersja w manifeście przebiegu | `63c171c4a3a4c8cfe818761f60033e815f43c4298368fdf93d0b5709469c7dc6`, 9 359 B | Wersjonowany plik pochodzi z `3d95f641288f57e5f7d4c9ed77c621b5f678daaf` (`feat(research): add BTC action value feasibility study`). Commit `6f5f55b3d7b0e60281d5ebd0a0793d8ed6bb853e` zawiera raport audytu, nie źródło tej wersji cech. | Ten sam kod bez końcowego pustego wiersza. |

Porównanie tekstu pokazuje wyłącznie różnicę jednego końcowego bajtu LF; zrzuty AST są identyczne. To różnica formatowania, nie semantyki. Wpis protokołu pozostaje nietknięty. Hash bieżącego pliku zgadza się z manifestem; brakujący commit zamrożonego blobu jest ograniczeniem pochodzenia, a nie nierozstrzygniętą różnicą zachowania.

## Pokrycie źródeł development

| Dane wymagane przez protokół | Zweryfikowane pokrycie lokalne | Brak / ograniczenie | Ocena możliwości uzupełnienia |
| --- | --- | --- | --- |
| Indeks rynków i oficjalne etykiety | Lokalny `pmxt_extended/market_index.parquet`: 21 962 rynki development, od `2026-04-15T17:05Z` do `2026-06-30T23:55Z`. Inne lokalne indeksy sprawdzone w audycie mają 0 wierszy przed tym startem. Wartości holdoutu w mieszanym pliku Parquet nie były materializowane. | Brak lokalnego indeksu przed 15 kwietnia. Dokumentacja Gamma pozwala stronicować wydarzenia/rynki, ale próbne zapytanie z tego środowiska nie rozwiązało DNS; kompletności historycznego zwrotu dla wcześniejszych BTC markets nie potwierdzono. | Endpoint metadanych istnieje, ale nie ma dowodu pokrycia historycznej populacji ani zweryfikowanego backfillu. |
| Zdarzenia `book`/`price_change` dla H | Archiwum `pmxt_extended` ma 1 853 pliki godzin development wobec 1 856 oczekiwanych: 1 409 niepustych i 444 potwierdzone puste. Brakujące godziny: `2026-06-11T04`, `T05`, `T06`; poprzedni audyt wykazał, że nie przecinają okien badanych rynków. Pierwsza godzina lokalnych eventów: `2026-04-14T16:00Z`; 25 godzin poprzedza pierwszy lokalny indeks rynku. Ciągłość źródła była kompletna dla 6 796 audytowanych lokalnych kandydatów. | Archiwum jest zewnętrznym źródłem, nie oficjalnym historycznym tape’em Polymarket; nie ma lokalnych zdarzeń przed pierwszą godziną. Same wcześniejsze eventy bez indeksu rynku nie tworzą pełnej populacji. | Brakujący plików nie uznano za przyczynę niedoboru — są poza badanymi oknami. Oficjalna dokumentacja opisuje publiczny bieżący order book, historyczne punkty ceny i strumień realtime `book`/`price_change`; nie udokumentowuje endpointu odtwarzającego historyczny pełny depth-event tape. Wniosek o braku takiego endpointu jest ograniczony do sprawdzonej dokumentacji. |
| T−60 i T−59 | Snapshoty/replay dependencies mają lokalne wiersze dla 21 962 development markets z tego samego lokalnego startu `2026-04-15T17:05Z`. Wcześniejsze lokalne pokrycie przed startem indeksu: 0. | Brak wcześniejszych drabin T−60/T−59. Aktualny `/book` zwraca stan bieżący i nie odtwarza historycznego stanu z chwili decyzji. | Nie ma lokalnego materiału do wcześniejszego odtworzenia ani dowodu czasów dostarczenia wcześniejszych snapshotów. |
| RAW OOF | Cały `B0_decision_oof.parquet`: 318 560 wierszy; `p_model_raw` skończone w 318 560. Zakres market startów: `2023-06-20T16:45Z`–`2026-06-30T23:50Z`; wszystkie wiersze są przed cutoffem. Podzbiór przed pierwszym lokalnym indeksem: 296 590 wierszy do `2026-04-15T17:00Z`. Każdy nominalny timestamp decyzji jest 60 s przed startem rynku. | `prediction_available_at_utc` jest pusty w 318 560/318 560 wierszy. Nominalny timestamp świecy nie dowodzi, kiedy dane dotarły ani kiedy predykcja była gotowa. | Wcześniejsze wiersze nie mogą spełnić wymogu dostępności T−59 bez historycznego pomiaru; nie dopisano domniemanych timestampów. |
| H, wszystkie zamrożone horyzonty | Poprawiony replay sprawdził wszystkie 21 H na 60 pierwotnie zaakceptowanych rynkach oraz na 4 odzyskanych rynkach; odpowiednio 1 260 i 84 wartości, bez rozbieżności. Po poprawce 2 397 lokalnych rynków ma wspólne S+H+RAW. | Nie istnieją dane odtwarzające H dla wcześniejszej populacji rynków, jeśli nie ma pełnych eventów book albo snapshotów początkowych. | Trzecia strona Polyfun reklamuje JSONL snapshotów BTC od lutego 2026, ale dostępny podgląd nie stanowi dowodu kompletności dla wszystkich tokenów/rynków ani tickowej ciągłości wymaganej przez H; nie włączono tego źródła do pokrycia i nie pobierano płatnego pełnego zbioru. |

Oficjalna dokumentacja Polymarket rozdziela listowanie metadanych rynków, bieżący order book i historyczne punkty ceny (`timestamp`, `price`) od realtime feedu z eventami `book`/`price_change`. To nie potwierdza historycznego odtworzenia głębokości księgi dla wcześniejszych T−60/T−59. Sprawdzono [Discover Markets](https://docs.polymarket.com/market-data/discover-markets), [Prices and Order Books](https://docs.polymarket.com/market-data/prices-order-books) i [Real-Time Data](https://docs.polymarket.com/market-data/realtime-data). Dodatkowym, nieoficjalnym kandydatem była [strona Polyfun](https://polyfun.space/); jej podgląd nie dowodzi wymaganej kompletności.

## Przyczynowość wcześniejszych RAW OOF i dostępność danych

- Modelowe OOF jest chronologiczne w obrębie 10 foldów i purge etykiet jest zapisany w manifestach. Nie czyni to jednak całej procedury walk-forward: źródłowe feature selection, wybór generatorów i tuning hiperparametrów były wykonane na pełnym development przed utworzeniem/wykorzystaniem wcześniejszych RAW OOF. Wcześniejsze predykcje są zatem warunkowe względem decyzji selekcyjnych, które widziały przyszłe fragmenty development. Dokumentacja źródłowego przebiegu również kwalifikuje wynik jako warunkowy względem wybranego zestawu cech i hiperparametrów.
- Późniejsza kalibracja dzienna używa wcześniejszych etykiet po regule dostępności/release i minimalnym progu liczebności, ale jest warunkowa względem tej samej selekcji. W dynamicznym przebiegu użyto surowego `p_model_raw`; kalibracja nie naprawia braku przyczynowości selekcji ani timestampów gotowości surowej predykcji.
- Kod eksportujący RAW OOF zapisuje `prediction_available_at_utc = NaT`; wszystkie 318 560 wierszy tak pozostało. W źródłowych świecach `actual_prediction_available_at` również jest `NaT`. Jednosekundowe opóźnienie to jawne założenie umowne, nie pomiar odbioru źródła, obliczenia cech i inferencji.
- Walidacja w runnerze dynamicznym porównuje `prediction_available_at_utc` z deadline’em, lecz porównanie `NaT > deadline` daje `False`; brak czasu nie blokuje wejścia. To luka integralności wejścia. Nie zmieniono zamrożonego runnera/protokołu ani nie uznano OOF za dostępne. Bez historycznych receive/compute timestamps nie da się dowieść, że dane i predykcja były dostępne przed T−60/T−59.

## Ochrona holdoutu w ścieżkach przygotowania danych

`btc_preopen_full_development_fit_20261016.json` wskazywał wcześniej plik `...research_cutoff_20261007.csv`. W bieżącym katalogu nie istnieje osobny development CSV `...development_cutoff_20260701.csv`. Nie odczytywano starego pliku ani nie tworzono z niego wycinka, bo wymagałoby to skanowania również holdoutu.

Dodano zabezpieczenia wymagające osobnego wejścia i granicy w UTC:

- runner full-development wybiera wyłącznie `development_raw_data_path` odrębne od źródła pełnej historii; brak tego pliku blokuje start przed fitami;
- `create_modeling_dataset.py` przy cutoff wymaga jawnego `development_input_path` zgodnego ze wskazanym plikiem oraz odrębnego `source_raw_data_path`; odrzuca świecę z timestampem `>= cutoff` i po budowie targetu odrzuca każdą skończoną etykietę, której market start jest poza development lub koniec horyzontu przekracza cutoff;
- runner pre-open sprawdza tę samą granicę przed generacją kontraktu i przekazuje timestamp końca etykiety do walidatora `fit_volume_profile.py`;
- `fit_volume_profile.load_base_ohlcv_frame` może przyjąć cutoff, odrzuca wiersze od cutoff oraz skończone targety z końcem horyzontu po cutoff.

Te zmiany nie tworzą osobnego CSV i nie odczytują żadnych dodatkowych danych. Wejście development pozostaje brakującym wymaganym artefaktem.

## Historyczny gate przed ścisłą kontrolą dostępności predykcji

Poniższa tabela dokumentuje wynik poprzedniego audytu księgi. Liczby 637, 908 i 1 207 są wyłącznie historyczną bazą przed odrzucaniem RAW OOF bez timestampu dostępności. Nie są wynikiem aktualnego gate’u; aktualny wynik znajduje się w addendum na końcu raportu.

Nie uzupełniono źródeł, zatem nie uruchomiono ponownie obliczenia gate’u. Zastosowanie tego samego frozen gate’u do ostatniego już skorygowanego audytu nadal daje **FAIL**:

| Warunek frozen gate | Ostatni wynik po poprawce audytowej | Minimum | Status |
| --- | ---: | ---: | --- |
| Kwalifikowane rynki ewaluacji | 637 | 2 000 | FAIL |
| Fold 1: treningowe rynki z q=$5 po obu stronach | 908 | 1 000 | FAIL |
| Fold 2: treningowe rynki z q=$5 po obu stronach | 1 207 | 1 000 | PASS |
| Dostępne oficjalne etykiety treningowe / fold | 1 760 / 2 279; obie klasy | 1 000 i obie klasy | PASS |
| Zaobserwowane dni / niepuste foldy | 22 / 2 | 20 / 2 | PASS |

Do minimum brakuje 1 363 rynków ewaluacji i 92 dwustronnych rynków q=$5 w treningu Foldu 1. To różnica względem progów, nie prognoza danych możliwych do odzyskania. Dodatkowe wcześniejsze rynki nie zmienią zamrożonego okresu ewaluacji; trzy brakujące godziny archiwum nie przecinają badanych okien; a obecne źródła nie dowodzą ani historycznej dostępności predykcji, ani pełnego event tape’u dla dodatkowych kandydatów. Nie zmieniono progów, foldów, feature order, definicji H ani reguł kwalifikacji.

Wynik końcowy: `blocked_missing_dynamic_history` oraz dodatkowy blok przyczynowości/dostępności RAW OOF. Aby wrócić do gate’u, potrzebne są: kompletny, źródłowo weryfikowalny indeks rynków i etykiet; pełne eventy native book od snapshotu inicjalizującego dla odpowiednich okien; zgodne T−60/T−59 drabiny; RAW OOF z pochodzeniem całego pipeline’u fold-local oraz rzeczywistym czasem dostępności wejść i predykcji; osobny development CSV z końcem targetów nie później niż zamrożony cutoff. Po otrzymaniu tych artefaktów należy ponownie zastosować istniejące reguły bez zmiany okresów i progów.

## Addendum — ścisła dostępność RAW OOF i wykonalna dalsza ścieżka

To addendum zastępuje wcześniejszy status gate’u. Protokół, daty, progi, foldy, 10 cech S, 21 cech H i reguły kwalifikacji pozostały bez zmian.

### Poprawka kontroli dostępności

W pierwotnym runnerze `NaT > deadline` zwracało `False`, więc brak czasu przechodził kontrolę. Runner teraz wymaga timestampu jawnie strefowego, normalizuje go do UTC i odrzuca `NaT`, tekst nieparsowalny oraz czas bez strefy jako `raw_prediction_availability_unproven`. Czas ściśle po deadline jest oddzielnym `raw_prediction_late`; równość jest dopuszczona zgodnie z freeze (`available_at <= T−59`, czyli T−60 plus 1 s). Wspólny replay i definicje cech nie zostały zmienione. Dodano pięć testów: NaT, błędny/czas bez strefy, konwersja do UTC, równość i przekroczenie deadline.

### Gate na zachowanych danych rozwoju

Skrypt [audit_btc_prediction_availability_20261020.py](../../audit_btc_prediction_availability_20261020.py) przeliczył gate na zachowanych artefaktach. Każdy skan Parquet zastosował warunek Arrow `market_start_utc < 2026-07-01T00:00:00Z` przed materializacją. Nie czytano etykiet ewaluacyjnych ani wierszy holdoutu.

| Okres | Kandydaci z audytu H | H zaliczone w poprzednim audycie | Dodatkowo odzyskane po niezależnym audycie | Nadal odrzucone przez historię/źródło | Bez dowodu dostępności RAW OOF | Spóźnione |
|---|---:|---:|---:|---:|---:|---:|
| Przed 2026-05-15 | 1 927 | 1 760 | 0 | 167 | 1 927 | 0 |
| Fold 1: 2026-05-15–2026-06-01 | 2 144 | 520 | 0 | 1 624 | 2 144 | 0 |
| Fold 2: 2026-06-01–2026-07-01 | 2 725 | 113 | 4 | 2 608 | 2 725 | 0 |
| **Razem** | **6 796** | **2 393** | **4** | **4 399** | **6 796** | **0** |

Brak dowodu dostępności nakłada się na pozostałe flagi H. Wśród tych samych 6 796 kandydatów pokrywa się z: `book_message_counted_as_reset` 4 357, `invalid_native_quote` 973, `equal_timestamp_or_ambiguous_event` 3 088, `equal_timestamp_price_conflict` 2 984, `ignored_same_side_stale_price_change` 246, `ignored_stale_book` 118 i `accepted_cross_side_late_price_change` 38. To liczniki unikalnych rynków z flagą w którymkolwiek horyzoncie; kategorie nakładają się i nie wolno ich sumować. Z 992 wcześniejszych głównych odrzuceń resetu księgi 4 zostały niezależnie odzyskane po sprawdzeniu identycznego pełnego stanu, a 988 nadal odpada. Każde z tych 992 ma również brak dowodu czasu predykcji.

| Zamrożony warunek gate’u | Poprzedni wynik księga-only — historyczny | Wynik po poprawce dostępności | Próg |
|---|---:|---:|---:|
| Rynki ewaluacji | 637 (520 / 117) | **0 (0 / 0)** | 2 000 |
| Dni ewaluacji | 22 | **0** | 20 |
| Niepuste foldy | 2 | **0** | 2 |
| Oficjalne etykiety z release przed pierwszą decyzją, fold 1 / 2 | 1 760 / 2 279 | 1 760 / 2 279 zapisanych; **0 / 0** z udowodnionym RAW on-time | 1 000 użytecznych wierszy i obie klasy w każdym foldzie |
| Znane q=$5 po obu stronach w treningu, fold 1 / 2 | 908 / 1 207 | 908 / 1 207 zapisanych; **0 / 0** z udowodnionym RAW on-time | 1 000 użytecznych rynków w każdym foldzie |

W lokalnym oficjalnym indeksie jest 21 962 rynków rozwoju. RAW OOF łączy się po starcie rynku z 21 961; wszystkie te wiersze mają dostępność nieudowodnioną, żaden nie jest spóźniony, a jeden start rynku — `2026-06-30T23:55:00Z` — nie ma RAW OOF. W całym pliku RAW OOF jest 318 560 skończonych predykcji, ale `prediction_available_at_utc` jest pusty w 318 560/318 560. Oficjalne etykiety nadal mają historyczne release counts 1 760 / 2 279, ale żaden nie może być użyty jako wiersz treningowy przy braku dowodu dostępności RAW cechy. Aktualny gate to **FAIL, 0 obserwacji kwalifikujących**. Liczb 637 i 908 nie należy przedstawiać jako poprawionego gate’u.

### Osobne blokady: historia księgi i przyczynowość predykcji

| Wymagane źródło | Pokrycie w development | Brak / status |
|---|---|---|
| Oficjalny indeks rynków | 21 962 starty, `2026-04-15T17:05Z`–`2026-06-30T23:55Z`; 21 961 ma dokładny join RAW OOF. | Brak lokalnego indeksu przed 15 kwietnia; jeden start 23:55 nie ma RAW. Pokrycie wcześniejszych oficjalnych rynków pozostaje niezweryfikowane. |
| Zdarzenia book/price-change dla H | 1 853/1 856 godzin rozwoju; 1 409 niepustych, 444 potwierdzone puste. Replay obejmuje lokalną populację od 15 kwietnia; niezależny verifier sprawdził próbę oraz 4 odzyskane okna. | 3 brakujące godziny 11 czerwca leżą poza audytowanymi oknami. 4 399 z 6 796 kandydatów nadal ma blokadę H/źródła; nie ma wcześniejszego pełnego tape’u. |
| Snapshoty T−60 / T−59 | Lokalne snapshoty i drabiny istnieją dla 21 962 development startów od 15 kwietnia. | Brak lokalnych snapshotów dla wcześniejszych startów. |
| RAW OOF | 318 560 skończonych `p_model_raw` do `2026-06-30T23:50Z`; wszystkie starty pliku są development. | Timestamp dostępności pusty 318 560/318 560; 21 961 złączonych lokalnych rynków ma brak dowodu; 1 lokalny market start nie ma wiersza OOF. |
| Wymagane H: 5 s, 15 s, 60 s | Replay wytworzył 21 cech na wspólnej lokalnej populacji; audyt niezależny porównał wszystkie 21 cech w 4 odzyskanych oknach oraz próbę oryginalnych akceptacji. | Dla kandydatów odrzuconych na źródle nie ma kompletu cech do zaliczenia; 4 399 nadal odrzuconych. |
| Wejście świec BTC do oddzielnego CSV rozwoju | Binance zwrócił próbki BTCUSDT i BTCUSD index na początku (2023-06-20) i tuż przed cutoffem; filtr `endTime=cutoff−1ms` nie zwrócił czasu holdoutu. | Pełnej ciągłości/parytetu źródła nie zweryfikowano. Nie pobierano archiwum. Nie jest to źródło Polymarket order book ani historycznych czasów inferencji. |

Zarejestrowana dostępność RAW OOF: **0**. Udowodniona rekonstrukcja: zapisane boostery potrafią odtworzyć istniejące wartości `p_model_raw`, a archiwum zdarzeń ma receive/source timestamps dla lokalnych okien, lecz to nie rekonstruuje historycznego czasu zakończenia inferencji. Dostępność wejść świec i gotowość predykcji dla starego procesu pozostają **nieznane**.

* **Historia księgi:** lokalne archiwum PMXT ma 1 853 z 1 856 oczekiwanych godzin rozwoju: 1 409 niepustych i 444 potwierdzone puste; brakujące 11 czerwca 04:00–06:00Z nie przecina sprawdzonych okien. Dla 6 796 lokalnych kandydatów zbadano ciągłość; po korekcie resetów księgi pozostaje 4 399 kandydatów poza wspólną kwalifikacją H/źródeł. T−60/T−59 i indeks rynku w lokalnych artefaktach zaczynają się dopiero 15 kwietnia 17:05Z. To blokada pokrycia źródłowego, odrębna od predykcji.
* **Przyczynowość i dostępność predykcji:** plik OOF zapisuje `prediction_available_at_utc = NaT`; kod eksportujący B0 wpisuje NaT jawnie. Nie ma historycznych czasów odbioru wszystkich wejść ani gotowości inferencji. Nie można wyliczyć ani dopisać ich z timestampu świecy; dlatego nawet odzyskana księga nie przywraca gate’u.

### Pochodzenie RAW OOF

| Element pipeline’u | Dowód / pochodzenie | Wiedza przyszła względem wcześniejszej predykcji? | Dostępność |
|---|---|---|---|
| B0 RAW OOF | 318 560 wyników; 10 chronologicznych fingerprintów foldów; zapisane boostery `models/B0/fold_00.txt`–`fold_09.txt`. Raport źródłowy podaje odtworzenie utrwalonych predykcji z boosterów bez ponownego fitu. | Sama walidacja i purge etykiet były fold-chronologiczne; to nie wystarcza do przyczynowości całej selekcji. | Wartości OOF można odtworzyć; czas gotowości historycznej jest nieznany. |
| Dane treningowe / etykieta proxy | 3 186 140 wierszy treningowych; cel `Close[Opened+6m] >= Open[Opened+2m]`. OOF używa decyzji na zamkniętych świecach, etykiety są purge’owane na granicy folda. | Etykiety treningowe w foldzie są wcześniejsze po purge; wynik własnej świecy staje się dostępny dopiero później. | Reguła etykiety i foldy są zapisane; faktyczny czas dotarcia świecy/etykiety do ówczesnego procesu nie jest zapisany. |
| Generatory i selekcja cech | Full-development pipeline wybrał generatory oraz 64 źródłowe cechy; późniejszy B0 dodaje sześć `taker_Z`/`taker_V` (70 łącznie). Raport full-development opisuje 17 ocen top-k i brak zewnętrznej pętli powtarzającej cały wybór. | **Tak.** Wybór na całym development do 30 czerwca patrzył poza wcześniejsze foldy. | Zarejestrowana selekcja; dla wcześniejszych predykcji warunkowa względem przyszłości. |
| Tuning i parametry | Parametry B0 pochodzą z pełnodevelopmentowego wyboru; raport źródłowy podaje stałe parametry dla 10-fold OOF. Raport wyboru opisuje 60 prób tuningu. | **Tak.** Późniejsze wiersze development uczestniczyły w wyborze użytym do wcześniejszych prognoz. | Parametry zachowane; brak fold-local rekonstrukcji całego wyboru. |
| Platt calibration | Dzienny Platt używa wyłącznie wcześniejszych oficjalnych etykiet OOF, których release wypada przed pierwszą decyzją dnia; wymaga 1 000 etykiet i obu klas. | Warunkowo nie, jeśli liczyć samą regułę kalibracji; wybór modelu pozostaje zanieczyszczony selekcją/tuningiem pełnego development. | Reguła i wyniki są zapisane. Dynamiczny gate używa `p_model_raw`, więc kalibracja nie naprawia RAW. |
| Dostępność wejść i predykcji | Wiersze źródłowe mają nominalne czasy świec; `actual_prediction_available_at` i `prediction_available_at_utc` są puste. | Nie da się wykazać, że komplet cech i predykcja były gotowe do T−59. | **Nieznana**; brak zarejestrowanego czasu i brak dowodu rekonstrukcji czasu. |

### Zachowane modele, koszt i rekomendowana ścieżka

Zachowano 10 boosterów B0 oraz konfigurację, które pozwalają odtworzyć zapisane conditional-OOF bez refitu. Nie są one jednak dowodem przyczynowości całego pipeline’u, bo wybór generatorów, 64 cech i parametrów korzystał z całego development. Zachowane walk-forward modele z innego eksperymentu mają inny zestaw/konfigurację cech i nie są zamiennikiem dla zamrożonego B0. Nie znaleziono artefaktu fold-local selekcji i tuningu B0 ani timestampów dostępności inferencji dla tych wierszy.

Najmniejszy sensowny refit źródła dla lokalnej populacji rozwoju miałby trzy punkty: seed wyuczony/wybrany wyłącznie na danych sprzed pierwszego lokalnego wspólnego rynku 15 kwietnia, następnie modele z prefiksów zamkniętych przed decyzjami foldów 15 maja i 1 czerwca. Selekcja i tuning muszą być zamrożone przed pierwszym takim forecastem; inaczej należy powtórzyć je fold-local. Podobny pełny pipeline miał przed biegiem kosztorys 5–8 h, a ukończony przebieg trwał aktywnie około 9 h 44 min z nieudaną próbą selekcji. Orientacyjny budżet to 5–8 h dla jednego wyboru przed pierwszym forecastem plus dwa dodatkowe refity; trzykrotne powtórzenie pełnego wyboru to około 15–24 h. To szacunek na podstawie innego przebiegu, nie benchmark tego OOF. Żadnego fitu tu nie uruchomiono.

**Rekomendacja:** przed jakimkolwiek kosztem refitu najpierw odzyskać oryginalne dzienniki inferencji/źródeł, jeśli istnieją, z rzeczywistymi receive i prediction-ready timestampami dla kwietnia–czerwca. W tym checkoutcie takich dowodów nie ma. Refity offline mogą wytworzyć przyczynowe wartości na podstawie prefiksów, ale nie odtworzą historycznego czasu gotowości; bez dzienników stary, zamrożony gate pozostanie zablokowany. Niezależnie można przygotować osobny CSV rozwoju z Binance: małe zapytania HTTP zwróciły po 4 wiersze dla BTCUSDT USDⓈ-M `/fapi/v1/klines` i BTCUSD index `/dapi/v1/indexPriceKlines` w 2023-06-20 16:45Z oraz 2026-06-30 23:56Z. Zapytanie kończyło się `endTime = cutoff − 1 ms`; ostatnia próbka zamknęła się przed cutoff. Pełnej ciągłości nie sprawdzano i nie pobierano całego archiwum. To pozwala zbudować osobne wejście świecowe z filtrem źródłowym, ale nie dostarcza Polymarket book history ani brakujących czasów RAW OOF. Zob. [Binance USDⓈ-M klines](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#kline-candlestick-data) oraz [Binance COIN-M index klines](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-coin-m-futures/api/rest-api/market-data#index-price-kline-candlestick-data). Oficjalny katalog Polymarket opisuje listowanie rynków, bieżącą księgę i realtime stream; nie wykazuje historycznego full-depth event tape: [Discover Markets](https://docs.polymarket.com/market-data/discover-markets), [Prices and Order Books](https://docs.polymarket.com/market-data/prices-order-books), [Real-Time Data](https://docs.polymarket.com/market-data/realtime-data).

### Commit 25a013be i kopie skryptów

Na wejściu HEAD był dokładnie `25a013be45ded28c0bfe9e93ff19d7149df25263`, bez zmian roboczych. Ten commit dodał ochronę cutoffu w ścieżce badawczej: `create_modeling_dataset.py` wymaga odrębnego `development_input_path` i `source_raw_data_path`, odrzuca wejście aliasujące pełną historię i sprawdza candle starty przed feature buildem; `features/btc_preopen_contract.py` odrzuca starty `>= cutoff` oraz etykietę kończącą się `> cutoff`; `fit_volume_profile.py` sprawdza te same granice przed fitowaniem VP; runner eksperymentu/full-development przekazuje osobny plik i cutoff. Równość końca etykiety z cutoff pozostaje dozwolona, jak w commitowym kodzie. Testy syntetyczne tych granic były już w commitcie.

Kopie `run.py` i `modeling.json` opisane jako stare w instrukcji załączonego dokumentu nie są bieżącą ścieżką badań: repozytoryjny `run.py` ostatnio zmienił się w `206cc713`, a ogólny `configs/modeling.json` w `c9938a33`; commit `25a013be` zmienia runner badawczy `run_btc_preopen_experiment.py` i wyspecjalizowany `configs/research/btc_preopen_full_development_fit_20261016.json`, nie live `run.py` ani ogólny config. W folderze załączników dostępne są instrukcje Markdown, nie osobne kopie tych dwóch skryptów, więc ich dokładnego bajtowego diffu nie da się wykazać. Nie użyto ich do nadpisania bieżącego repo. Zabezpieczenia chronią wskazane ścieżki badawcze; nie zmieniono ścieżki live.

Rozbieżność hashy pliku cech z poprzedniej części raportu pozostaje czysto bajtowa: zamrożony hash `f7d6…` wskazuje osierocony blob 9 360 B z dodatkowym końcowym LF; bieżący plik 9 359 B z commita `3d95f641…` ma identyczny AST i hash zgodny z manifestem. Commit `25a013be` nie zmienił definicji cech ani nie edytował freeze.

### Ochrona cutoffu i walidacja tej aktualizacji

Wywołane testy: `python -m unittest tests.test_btc_market_dynamics_action_value_20261020 tests.test_btc_preopen_contract tests.test_btc_development_boundary` — 38 PASS. Sprawdzają dostępność, granice danych i horyzont etykiety; skrypt gate’u nie fituje modeli ani nie liczy ekonomiki. Stary frozen protocol, jego hash, folder źródłowego przebiegu i holdout pozostawiono nietknięte.

## Walidacja zmian kodu

Na sztucznych danych sprawdzono ochronę wejścia i końca etykiety oraz zgodność ścieżki osobnego pliku. `python -m unittest tests.test_btc_preopen_contract tests.test_btc_development_boundary tests.test_btc_preopen_full_development_fit`: **24 testy, PASS**. `git diff --check`: PASS. Żadnego treningu, tuningu, replayu archiwum ani ekonomiki nie uruchamiano.
