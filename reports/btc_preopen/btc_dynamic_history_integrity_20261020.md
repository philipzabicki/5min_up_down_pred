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

## Gate i decyzja

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

## Walidacja zmian kodu

Na sztucznych danych sprawdzono ochronę wejścia i końca etykiety oraz zgodność ścieżki osobnego pliku. `python -m unittest tests.test_btc_preopen_contract tests.test_btc_development_boundary tests.test_btc_preopen_full_development_fit`: **24 testy, PASS**. `git diff --check`: PASS. Żadnego treningu, tuningu, replayu archiwum ani ekonomiki nie uruchamiano.
