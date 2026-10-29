# E0043 — uzupelnienie zrodel 68-godzinnej luki ksiegi

Data audytu: 2026-10-29. Zakres UTC: `[2026-08-15T10:00:00Z, 2026-08-18T06:00:00Z)` — dokladnie 68 godzin. Protokol: [E0043 v2](../../configs/research/btc_D_holdout_book_gap_fill_20261029_v2.json); wersja v1 pozostaje zachowana.

## Wynik

| Pytanie | Wynik |
|---|---|
| Dostepnosc i pobranie | 0/68 godzin z kwalifikujacym, pelnym venue-native event tape; 0 partycji pobranych lub sprawdzonych payloadowo. Wszystkie 68 godzin sprawdzono metadanymi. |
| Kompletne okna z 1 115 | 0. Nadal niekompletne: 816 bez metadanych entry-hour/freshness oraz 299 z entry-hour, ale z niepelna wczesniejsza historia 25/26 h. |
| Dwa niedopasowane sloty | `2026-07-12T02:25Z` i `2026-09-03T15:20Z` pozostaja nierozstrzygniete: brak wierszy dla obu startow w lokalnych technicznych indeksach Gamma i PMXT. |
| Zgodnosc cache cech | Nie. Szeroki cache nie zgadza sie z parametrami RP/VP E0041; sama zgodnosc nazw kolumn nie potwierdza zgodnosci wartosci. Przyszly builder musi odtworzyc cechy z surowych wejsc i zamrozonych transformacji. |
| Przygotowanie wejsc | Zablokowane brakiem pelnej natywnej ksiegi dla `[2026-08-15T10:00Z, 2026-08-18T06:00Z)`. Nie mozna potwierdzic T-60/T-59, inicjalizacji, BBO/freshness ani source-quality gate. |

## Zdalne zrodla i dowod niedostepnosci

Publiczne indeksy wydawcy PMXT V1, PMXT V2, AG6 V2 i PendulumFlow V3 zwrocily HTTP 200, ale kazdy zawieral **0 obiektow Parquet** z docelowych 68 godzin. SHA-256 pobranych indeksow oraz czas kontroli znajduja sie w [maszynowym protokole zrodel](btc_D_holdout_book_gap_fill_20261029_sources.json). PMXT V1 konczy sie w kwietniu 2026, wiec nie obejmuje badanego okresu ([indeks PMXT V1](https://archive.pendulumflow.com/browse/pmxt/v1/)).

Dokumentacja PMXT V2 opisuje pliki godzinowe i mirror konczacy sie 2026-08-09T23; oryginalny PMXT ma ostatnia wskazana godzine 2026-08-10T00. Zgodnie z udokumentowanym wzorcem URL wykonano HEAD dla wszystkich 68 plikow docelowych — kazdy zwrocil 404. Obsluge HEAD potwierdzono odpowiedzia 200 dla obiektu kontrolnego poza zakresem; nie odczytano jego body i nie uzyto go do pokrycia. Dokumentacja oraz odpowiedzi sa zapisane w [protokole zrodel](btc_D_holdout_book_gap_fill_20261029_sources.json) i [przegladzie danych PMXT V2](https://archive.pendulumflow.com/formats/v2).

Indeks AG6 V2 rowniez nie zawiera docelowych godzin. Strona archiwum jawnie opisuje venue-wide gap od `2026-08-15T10:00Z` do `2026-08-18T06:00Z`; V3 zaczyna sie o `2026-08-18T06:00Z` ([indeks AG6](https://archive.pendulumflow.com/third-party/ag6/), [przeglad V3](https://archive.pendulumflow.com/formats/v3)). Jest to konkretne potwierdzenie luki wydawcy, a nie wniosek oparty wylacznie na lokalnym braku plikow.

Sprawdzona lista kontrybutora devpyle ma 4 pliki na kazda z 68 godzin (272 pliki, deklarowane rozmiary po zaokragleniu razem 8 692,06 MiB). Opis formatu ogranicza `clob_l2` do pieciu poziomow, dotyczy wlasnego formatu rynkow crypto Up/Down i podaje czas odbioru rejestratora bez czasu venue. Nie jest to pelna natywna ksiega z wymagana chronologia, wiec zrodlo nie kwalifikuje sie; plikow nie pobrano ([lista](https://archive.pendulumflow.com/contrib/devpyle/), [audyt formatu](https://archive.pendulumflow.com/audit/devpyle)).

Brak pelnego obiektu oznacza brak rozmiaru i checksumy dla kwalifikujacych partycji. Zatem transfer kwalifikujacych danych wyniosl 0 B, staging nie powstal, a nowe partycje eventow nie zostaly zapisane. Deklarowane 8,49 GiB kontrybucji nie stanowilo przeszkody zasobowej: odrzucono ja ze wzgledu na semantyke i zakres danych. Nie dodano download-era ani parsera, bo nie ustalono dostepnego kwalifikujacego obiektu.

## Ponowne wyliczenie 1 115 okien

Nakladke wyliczono na istniejacym technicznym `market_index.parquet`, uzywajac tylko wymienionych kolumn identyfikacji rynku i archiwalnej historii. Dla kazdego okna zastosowano regule `archive_window_hours_processed == archive_window_hours_expected`; nie ustawiano recznie flag kompletnosci. Nie odczytano kolumn etykiet, payoutow, predykcji ani wartosci cech.

| Grupa | Liczba | Wynik |
|---|---:|---|
| Starty `2026-08-15T10:05Z`–`2026-08-18T06:00Z`: brak entry-hour/freshness metadata | 816 | Nadal niekompletne; brak kwalifikujacych godzin do dodania |
| Starty `2026-08-18T06:05Z`–`2026-08-19T06:55Z`: entry-hour jest, brakuje wczesniejszego okna | 299 | Nadal niekompletne; oczekiwane okno 25/26 h nie jest pokryte |
| Razem, bez podwojnego liczenia | 1 115 | 0 stalo sie kompletnych; wszystkie 1 115 pozostaja niekompletne |

Kazde z tych 1 115 okien przecina sie z badana luka w co najmniej jednej i najwyzej 26 godzinach. Szczegoly per rynek sa w [nakladce CSV](btc_D_holdout_book_gap_fill_20261029_coverage.csv), a podsumowanie i hash nakladki w [protokole JSON](btc_D_holdout_book_gap_fill_20261029_sources.json). Nie potwierdzono pokrycia cutoffow T-60/T-59 ani wymaganego startu inicjalizacji; nie wykonano replayu.

## Niedopasowane sloty

Dla Gamma i PMXT odczytano tylko techniczne kolumny `market_start_utc`, `condition_id`, `market_slug`, `up_token_id` i `down_token_id`. Zaden indeks nie ma wiersza dla zadnego z dwoch startow. Nie odczytywano wynikow ani payoutow. Sloty pozostaja osobnym, nierozstrzygnietym problemem i nie zostaly doliczone do 1 115. Kolejny bezpieczny krok identyfikacyjny wymaga dostepnej projekcji publicznych lub lokalnych metadanych condition/slug/token dla tych dwoch czasow.

## Tozsamosc cache cech i przyszly builder

Szeroki cache zawiera przez schemat 64/64 wybranych bazowych cech E0041; sprawdzenie schematu nie odczytalo wartosci. Konfiguracje RP maja rozne SHA-256 (`961cbcfb...` cache, `c645a957...` E0041), m.in. rozne liczby binow 214 334 i 29 322. Konfiguracje VP rowniez maja rozne SHA-256 (`1f889a7f...` cache, `e53cf3c0...` E0041), m.in. `neighbor_bins=6` i `neighbor_bins=5`. Wersje RP/VP nie dowodza zgodnosci parametrow. Cache nie zawiera hashy generatorow cech wskaznikowych.

Szesc cech `taker_Z/V` dla horyzontow 1/5/15 minut pochodzi z osobnego generatora i osobnego produktu. Znane definicje korzystaja m.in. z przeplywu quote-volume, zwrotu logarytmicznego i historycznych skal zmiennosci/obrotu; hash kodu generatora zachowany w metadanych walidacji to `127a0a53...`. E0041 pin-uje bajty istniejacego prefiksu przeplywow (`8a9dc8d1...`), ale hash generatora nie znajduje sie w jego liscie `implementation_hashes`. To nie wystarcza do potwierdzenia, ze obecny szeroki cache odtworzy zamrozona transformacje. Pelne wartosci SHA i parametry sa zapisane w [protokole JSON](btc_D_holdout_book_gap_fill_20261029_sources.json).

Przyszly builder powinien tworzyc cechy z dokladnych surowych wejsc oraz transformacji wskazanych kodem E0041. Stan pamieci/ESN nalezy rozgrzac chronologicznie z raw/canonical pipeline; koncowe PCA z dowolnego Parquet nie odtwarza tego stanu. W tym zadaniu nie generowano cech ani prognoz.

## Stan ochrony i odchylenie proceduralne

E0041 pozostaje `D_BUNDLE_READY`; SHA-256 manifestu ponownie potwierdzono: `7715419ce5efbce8405db861fb3a67c481038c07f11eec019137f14528f0f97d`. `evaluation_authorized=false`, `activated=false`, `HOLDOUT_NOT_RUN`; 0 prognoz, 0 metryk predykcyjnych i 0 obliczen ekonomicznych. Nie zmieniono D, modeli, cech, kalibratora, cutoffow ani opoznien. T0046/T0048 pozostaja nietkniete; E0037 pozostaje FAILED, T0009 BLOCKED, T0010 DEFERRED.

Odnotowane odchylenie: wykonano jeden metadanymi-only HEAD dla `2026-08-10T00:00Z`, poza zakresem, zeby potwierdzic obsluge metody HEAD. Odpowiedz 200 zawierala metadane obiektu; body nie odczytano, a obiektu nie uzyto do pokrycia. Pelny rejestr tego zdarzenia jest w JSON. Nie pobrano zadnych danych holdoutowych ani nie uruchomiono live/shadow lub zlecen.
