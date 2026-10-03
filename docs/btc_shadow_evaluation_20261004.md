# Aneks do oceny shadow BTC — 2026-10-04

Ten aneks dotyczy diagnostyki po commitach `1941e3a` i `05812025`. Zachowuje raport i protokół v1 bez zmian. Nowa sesja v2 ma osobny config, manifest i bazę; używa tego samego modelu BTC oraz tej samej polityki wejścia. Wszystkie fill’e i opłaty nadal są symulowane.

## 1. Nowy harmonogram i uzasadnienie starego

Stary próg 730 dni wynikał z częstotliwości transakcji: wcześniejszy shadow miał 5 hipotetycznych transakcji `market_plus_btc` w około 32,7 dnia, czyli 0,153 dziennie. Przy tej częstości 100 transakcji wymaga około 655 dni; 730 dni dawało prognozowane 112 transakcji. Limit 1 095 dni był wyłącznie górnym limitem na wypadek, gdyby do dnia 730 nie było 100 transakcji — nie wynikał z analizy mocy.

Próbka prognoz ma inną skalę. W zbiorze developerskim było 9 407 wspólnych okien z 32,7 dnia. Dla `market_plus_btc − market_only` raportowano Δ log loss −0,000235 (95% CI [−0,000464, −0,000064]) i Δ Brier −0,000121 ([−0,000231, −0,000035]). Przedziały pochodziły ze sparowanego moving-block bootstrapu bloków kalendarzowych po 3 dni, 2 000 replikacji. Przybliżenie normalne z szerokości przedziału log loss daje około 13 900 okien dla 80% mocy, czyli około 48–50 dni przy 288 oknach dziennie; Brier daje nieco mniej, więc log loss ogranicza tę estymację.

To wyłącznie orientacyjny rachunek z próby, która uczestniczyła w wyborze modelu. Zakłada podobny efekt i rozkład w przyszłości. Bloki 3-dniowe uwzględniają część autokorelacji, ale mogą nie obejmować dłuższych reżimów; niepewność co do efektu, stacjonarności i kosztów handlu pozostaje. Próbka 100 transakcji nie jest dowodem mocy dla rentowności: to jedynie dawny minimalny count. Przy 30 dniach obecna częstość implikuje około 4–5 transakcji, a nie 100.

Protokół v2 zastępuje wieloletni harmonogram trzema kontrolami liczonymi od startu v2:

- 48 godzin: stan procesu, ciągłość świec, deduplikacja i zaległe rozstrzygnięcia;
- 7 dni: jakość danych i częstość sygnałów, bez wniosku o przewadze;
- 30 dni: raport jakości prognoz i symulowanej ekonomiki, a następnie jawna decyzja, czy warto kontynuować.

Po 30 dniach dopuszczalny wniosek to „nierozstrzygnięte; obecna częstość transakcji nie uzasadnia dalszego testowania tej wersji”. Konfiguracja nie przedłuża testu automatycznie. Datowany config v2: [protokół](../configs/btc_shadow_protocol_20261004.json). Oryginalny [protokół v1](../configs/btc_shadow_protocol_20261003.json) pozostaje nietknięty.

## 2. Wszystkie sygnały i ograniczenie gotówki

Odtworzenie stałej polityki na historycznych danych dało 810 kwalifikujących się sygnałów `btc_platt`. Dla diagnostyki każdego liczono osobne hipotetyczne wejście za 5 USD, bez limitu gotówki portfela: 4 050 USD obrotu, 166,34 USD symulowanych opłat, deklarowane EV +776,997 USD, zrealizowane PnL −383,72 USD po kosztach, 318/810 wygranych (39,26%). To nie jest portfel ani wykonalny bilans.

Replay portfelowy z początkową gotówką 100 USD zawarł 73 transakcje: 24/73 wygrane, 14,70 USD opłat, PnL −97,89 USD i 2,11 USD końcowej gotówki. Pozostałe **737 sygnałów** odrzucono wyłącznie z powodu `insufficient_cash`; nie znikają z analizy. Wśród nich 294/737 (39,89%) wygrały, a niezależne hipotetyczne wejścia po 5 USD dałyby −285,84 USD. To nadal diagnostyka sygnałów, bez ograniczenia bankrollu.

Każdy rekord sygnału zachowuje wybraną stronę, p modelu i rynku, ask wykonania, próg rentowności po opłacie, przewidywane EV, oficjalny wynik, dostępność kwotowania oraz jego wiek lub jawny brak tej informacji. Plik wynikowy jest lokalnym artefaktem: `data/analysis/polymarket/BTC/entry_selection_audit_20261003/all_independent_btc_platt_entry_signals_1s.parquet`. Skrypt odtwarzający: [audit_btc_entry_selection.py](../audit_btc_entry_selection.py). Hipotetyczne wykonanie jest liczone po widocznym asku i opłacie z zamrożonego modelu 7%; rzeczywisty fill i opłata są nieznane.

W całych 810 sygnałach średnie p modelu wybranej strony wyniosło 50,60%, p rynku 41,28%, a trafność 39,26%. Średni ask wyniósł 0,413; średni próg rentowności po kosztach 43,08%. Model deklarował więc przewagę 7,53 p.p. ponad próg, której nie potwierdziły wyniki. Dla UP: 178/463 wygranych (38,44%), model 50,91%, rynek 41,47%, PnL −283,18 USD. Dla DOWN: 140/347 (40,35%), model 50,20%, rynek 41,02%, PnL −100,54 USD.

Przedziały ustalono przed agregacją: p rynku `<0,35`, `0,35–0,45`, `≥0,45` oraz luka `p_model − p_market` `<0,05`, `0,05–0,10`, `≥0,10`. Foldy są trzema chronologicznymi okresami: 15–26 kwietnia, 26 kwietnia–7 maja i 7–18 maja 2026. PnL to suma niezależnych wejść po 5 USD po kosztach; grupy są małe i opisowe, a dane należały do okresu developerskiego.

### Przedziały prawdopodobieństwa rynku

| Fold | Strona | p rynku | n | Śr. p modelu | Śr. p rynku | Wygrane | PnL po kosztach |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | DOWN | <0,35 | 5 | 48,61% | 32,50% | 40,00% | +$2,66 |
| 0 | DOWN | 0,35–0,45 | 72 | 49,14% | 40,94% | 37,50% | −$42,80 |
| 0 | DOWN | ≥0,45 | 31 | 52,50% | 46,66% | 48,39% | +$2,47 |
| 0 | UP | <0,35 | 5 | 50,61% | 33,70% | 40,00% | +$3,05 |
| 0 | UP | 0,35–0,45 | 84 | 49,98% | 41,10% | 30,95% | −$120,01 |
| 0 | UP | ≥0,45 | 43 | 52,81% | 46,91% | 48,84% | −$2,60 |
| 1 | DOWN | <0,35 | 8 | 49,54% | 32,88% | 50,00% | +$18,67 |
| 1 | DOWN | 0,35–0,45 | 58 | 49,47% | 40,67% | 36,21% | −$43,92 |
| 1 | DOWN | ≥0,45 | 27 | 51,99% | 46,28% | 59,26% | +$30,84 |
| 1 | UP | <0,35 | 11 | 50,39% | 31,68% | 18,18% | −$25,62 |
| 1 | UP | 0,35–0,45 | 71 | 50,08% | 41,01% | 47,89% | +$38,58 |
| 1 | UP | ≥0,45 | 59 | 52,00% | 46,53% | 40,68% | −$46,68 |
| 2 | DOWN | <0,35 | 35 | 49,68% | 32,61% | 31,43% | −$12,65 |
| 2 | DOWN | 0,35–0,45 | 82 | 49,59% | 40,32% | 37,80% | −$42,23 |
| 2 | DOWN | ≥0,45 | 29 | 52,92% | 46,83% | 44,83% | −$13,59 |
| 2 | UP | <0,35 | 52 | 50,01% | 32,66% | 25,00% | −$65,64 |
| 2 | UP | 0,35–0,45 | 88 | 50,53% | 40,11% | 36,36% | −$63,31 |
| 2 | UP | ≥0,45 | 50 | 52,49% | 46,61% | 48,00% | −$0,95 |

### Przedziały luki model–rynek

| Fold | Strona | Luka p model − p rynku | n | Śr. p modelu | Śr. p rynku | Wygrane | PnL po kosztach |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | DOWN | <0,05 | 8 | 49,57% | 44,75% | 25,00% | −$18,62 |
| 0 | DOWN | 0,05–0,10 | 76 | 50,08% | 43,62% | 46,05% | +$9,02 |
| 0 | DOWN | ≥0,10 | 24 | 50,24% | 36,83% | 29,17% | −$28,06 |
| 0 | UP | <0,05 | 10 | 50,14% | 45,30% | 10,00% | −$39,96 |
| 0 | UP | 0,05–0,10 | 90 | 51,03% | 44,20% | 43,33% | −$34,23 |
| 0 | UP | ≥0,10 | 32 | 50,89% | 37,72% | 28,12% | −$45,36 |
| 1 | DOWN | <0,05 | 7 | 49,00% | 44,21% | 57,14% | +$9,25 |
| 1 | DOWN | 0,05–0,10 | 58 | 50,36% | 43,76% | 48,28% | +$14,93 |
| 1 | DOWN | ≥0,10 | 28 | 50,20% | 36,57% | 32,14% | −$18,59 |
| 1 | UP | <0,05 | 22 | 50,91% | 46,05% | 45,45% | −$6,65 |
| 1 | UP | 0,05–0,10 | 83 | 50,89% | 44,49% | 45,78% | +$0,83 |
| 1 | UP | ≥0,10 | 36 | 50,95% | 36,11% | 33,33% | −$27,90 |
| 2 | DOWN | <0,05 | 16 | 49,37% | 44,56% | 31,25% | −$26,86 |
| 2 | DOWN | 0,05–0,10 | 63 | 50,56% | 43,59% | 42,86% | −$19,04 |
| 2 | DOWN | ≥0,10 | 67 | 50,22% | 35,02% | 34,33% | −$22,57 |
| 2 | UP | <0,05 | 12 | 50,52% | 45,67% | 50,00% | +$5,03 |
| 2 | UP | 0,05–0,10 | 82 | 51,31% | 44,57% | 41,46% | −$48,83 |
| 2 | UP | ≥0,10 | 96 | 50,60% | 34,96% | 30,21% | −$86,11 |

Przy luce co najmniej 10 p.p. wszystkie sześć kombinacji fold–strona miało stratę. Łącznie było 283 sygnałów, model średnio około 50,5%, rynek około 35–38%, trafność 31,45% i PnL −228,59 USD. To wspiera hipotezę, że polityka kupowała stronę tanią według rynku, bo model trzymał się blisko 50%, mimo że rynek w tych wybranych sygnałach trafiał lepiej. Nie dowodzi to przyczynowości ani przewagi rynku poza tą selekcją. Tanie UP `<0,35` przyniosły −88,21 USD (68 sygnałów); tanie DOWN +8,69 USD (48 sygnałów), więc nie ma podstaw do automatycznego odwracania strony.

## 3. Zgodność chwili prognozy z kwotowaniem

W historycznym replayu wejściowy BTC to jednominutowa świeca zamknięta na początku pięciominutowego rynku: `Opened = market_start − 1 min`; jej dane są dostępne dopiero po zamknięciu. Historyczna ocena zakładała dostępność prognozy w `market_start`, bez rzeczywistej latencji feedu i obliczeń. Pierwsze kwotowanie użyte do p rynku przypadało na `market_start`; cena hipotetycznego wykonania była ask z `market_start + 1 s`. W 810 kwalifikujących się wejściach p rynku wybranej strony zmieniło się od startu do kwotowania wykonania średnio o −0,00529 (−0,53 p.p.; SD 1,51 p.p.; mediana 0; zakres −11 do +4 p.p.). Wiek książki na historycznych snapshotach jest nieznany; częstotliwość próbkowania nie jest miarą wieku. Dostępna historyczna świeca BTC ma minutową, a nie sekundową rozdzielczość, więc nie pozwala zmierzyć sekundowego ruchu BTC po starcie.

Collector v1 mierzy czasy ścienne: świeca BTC jest odbierana, inferencja zaczyna się i kończy, następnie pobierane są UP/DOWN książki i zapisywana decyzja. Te pola, łącznie z wiekiem znacznika książki i lokalnym czasem odbioru, są zapisane osobno. Znacznik zewnętrznej książki nie dowodzi jej wieku bez znanego offsetu zegarów. W szczególności przyszły względem lokalnego odbioru timestamp jest traktowany jako nieważny, a nie jako świeży kwot.

Polymarket Gamma w zapisanym payloadzie wskazuje rozstrzygnięcie BTC 5m na podstawie 60-sekundowego TWAP BTC/USD z Chainlink: UP, gdy TWAP jest co najmniej ceną z początku zakresu, w przeciwnym razie DOWN. Historyczny target pochodzi z oficjalnego rozstrzygnięcia rynku; minutowa Binance cena nie zastępuje tego źródła.

## 4. Rozbieżność offline/live cechy i poprawka

Rozbieżna cecha to `ChaikinOsc_fit_1440m_pop128_fasmatypt3_fasper757_slomatypgma_sloper978_qe0.1_qm0.2_tf0.8_stmc_sg15`. `1440m` w nazwie jest horyzontem targetu, nie resamplingiem. Na 2026-10-02 04:54 UTC zapis offline wynosił −712,3093286585, a live −622,6774851059: różnica 89,6318435526, czyli 12,58% wartości offline. W tym 288-punktowym oknie mediana |offline| wyniosła 404,07, a 95. percentyl 1 480,96. Izolowana zamiana tej cechy w tym jednym wierszu nie zmieniła p modelu (0,47806095). W szerszym replayu 4 032 decyzji maksimum wyniosło 433,9933398288 (offline 37,4109207648; live 471,4042605936). W 77/4 032 oknach live wybierał inną gałąź GMA zamiast SMA.

Przyczyną jest wybór algorytmu zależny od zakresu historii, nie resampling, precyzja ani brak świecy. Offline liczył Chaikin ADL po pełnej historii; w niej występuje wartość nie-dodatnia, więc `GMA_or_SMA` wybiera SMA dla wolnej średniej. W 21 936-świecowym oknie live rebazowany ADL bywał cały dodatni, przez co ta sama funkcja wybierała GMA. W takim oknie oba obliczenia dostają identyczne OHLCV, ale inną średnią. Cecha ma 1 154 wystąpienia w podziałach LightGBM; przed korektą w szerokim replayu 32/4 032 wiersze przekraczały co najmniej jeden próg tej cechy (1 671 przekroczeń węzeł–wiersz).

Runtime v2 jawnie wymusza dla tej jednej, zidentyfikowanej cechy `slow_ma_type=SMA`, gdy skonfigurowany fit oczekuje `GMA`; pozostałe parametry, model i polityka pozostają zamrożone. W niezależnym przeliczeniu tych samych 4 032 okien skorygowana cecha miała maksymalną różnicę 1,02e−7 względem offline; żadne okno nie przekroczyło 1e−6. W drzewach było 1 154 splitów na tej cesze; przed poprawką 32/4 032 wiersze przekraczały co najmniej jeden próg (1 671 przekroczeń węzeł–wiersz), po poprawce nie przekroczył go żaden wiersz.

Pełny replay po poprawce: 20 160 świec, 4 032 decyzje, 256 cech. Parzystość cech i predykcji jest raportowana osobno w `data/analysis/live_feature_parity/BTC/20261004_sma_override/live_vs_stored_summary.json`; wskazana cecha Chaikin jest zgodna w podanym limicie. Pozostało 5/4 032 różnic prawdopodobieństwa powyżej 1e−6 (maks. 0,00040394), wyjaśnionych inną cechą ADX; nie było żadnego odwrócenia sygnału UP/DOWN. W rekordach audit decyzja ma jawny status `not_verified_missing_quotes`, ponieważ nie ma historycznych kwotowań. Testy osobno obejmują parzystość cechy, raport parzystości predykcji, status decyzji, restart i catch-up po luce.

Oryginalne wymagania runtime zachowano w `data/runtime/BTC/indicator_history_requirements_20261003.json`; wersja z override ma osobną ścieżkę i hash.

## 5. Stan collectora, trwałość i jedna następna próba

V1 używa publicznych GET i zapisuje świecę, decyzje, rozstrzygnięcia oraz ledger symulacyjny do SQLite. Klucze unikalne i `INSERT OR IGNORE` chronią duplikaty; rozstrzygnięcie jest osobnym, dopisywanym zdarzeniem, więc backfill nie mutuje zapisanej decyzji. Rynek zakończony, wynik oficjalny dostępny i gotówka symulacyjna dostępna to osobne stany. Start/wznowienie odtworzenia z checkpointu lub zachowanych świec oraz REST catch-up po luce sprawdza test; brak świecy zatrzymuje catch-up zamiast po cichu przesuwać stan.

**Następny eksperyment: aktualizować prawdopodobieństwo na moment wejścia.** Historyczne p modelu było symulowane przy starcie rynku, a ask wykonania pochodził sekundę później; w collectorze dostępność wejść i kwotowań pojawia się już w trakcie okna. Jednocześnie wybrane wejścia z dużą luką model–rynek miały prognozę modelu bliską 50%, p rynku znacznie niższe i słaby wynik w obu stronach. Następna ograniczona próba powinna więc mierzyć BTC i Polymarket współcześnie w chwili rzeczywistej decyzji, a potem ocenić prognozę właśnie dla tej chwili. Nie odwracamy sygnałów ani nie uruchamiamy strojenia progów.


## 6. Stan po uruchomieniu v2 i rotacji collectora

Sesja v2 rozpoczęła się **2026-10-03 23:13:15.016517 UTC** jako `btc_new_model_shadow_v2_20261004`. Collector v2 odbudował historię REST od zapisanego punktu wznowienia i zapisał ciągłe świece od `2026-10-01 18:00 UTC`; podczas rotacji obie bazy miały identyczny zakres do `2026-10-03 23:20 UTC`, 3 201 unikalnych minut i zero luk. W v2 zapisano pierwsze dwa rynki (6 decyzji: każdy z trzech wariantów), bez błędów collectora. Hash manifestu obejmuje 112 artefaktów; `trading_enabled=false`, `order_submission_count=0`, a log potwierdza `orders=disabled`.

Po potwierdzeniu catch-up, manifestu i pierwszych decyzji zatrzymano wyłącznie stary proces v1, PID 27724, o **2026-10-03 23:21:38.704613 UTC**. Jego baza jest zachowana osobno i nie jest już modyfikowana przez proces. Przy zatrzymaniu zawierała 37 rynków / 111 decyzji, po jednym komplecie trzech wariantów na rynek, 3 201 ciągłych świec, 35 rozstrzygnięć, 216 unikalnych zdarzeń ledgeru oraz kontrolę SQLite `integrity_check=ok`. Dwa ostatnie rynki nie miały jeszcze rozstrzygnięcia: rynek z 23:15 UTC zakończył się, lecz wynik nie był jeszcze dostępny collectorowi; rynek z 23:20 UTC nadal trwał. Późniejszy wynik dopisuje osobny rekord settlement i nie zmienia decyzji. Stan środków symulowanych w ledgerze jest odrębny od zamknięcia rynku i dostępności wyniku.

V2 pozostał uruchomiony w tle jako PID 19764. Przy tej samej migawce miał 2 rynki, 6 decyzji, 6 unikalnych zdarzeń ledgeru i zero settlementów; oba wyniki oczekiwały na publikację. Nadal nie są wysyłane zlecenia. Gdy oficjalny wynik pojawi się w źródle, v2 zapisze go append-only; brak wyniku na tej migawce nie oznacza wyniku przegranego ani braku środków symulowanych.


Dalsza kontrola v2 o **2026-10-03 23:26:35 UTC** potwierdziła dopisanie pierwszego wyniku po jego publikacji: dla rynku rozpoczętego o 23:15 UTC Gamma podało `resolved_at=23:22:31`, a collector zaobserwował i dopisał settlement o 23:25:16. Decyzje pozostały osobnymi rekordami; baza miała wtedy 3 rynki, 9 decyzji (3 warianty na rynek), 1 settlement i 12 unikalnych zdarzeń ledgeru. Było 3 206 ciągłych świec do 23:25 UTC, `integrity_check=ok`, bez błędów collectora; PID 19764 nadal działał, `trading_enabled=false`, `order_submission_count=0`. Dwa pozostałe rynki oczekiwały na rozstrzygnięcie.

Punkty oceny v2, liczone od czasu startu, przypadają na **2026-10-05 23:13:15.016517 UTC (48 h)**, **2026-10-10 23:13:15.016517 UTC (7 dni)** i **2026-11-02 23:13:15.016517 UTC (30 dni)**. Nie ma automatycznego przedłużenia po ostatnim punkcie.
