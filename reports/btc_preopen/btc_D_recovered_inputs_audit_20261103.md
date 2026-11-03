# E0048 — audyt odzyskanych wejść D

**Data audytu:** 2026-11-03

**Punkt odniesienia:** commit 191766c9a998e3d9907ffda1c7ac06fed7b2a622, branch audit/btc-dynamic-history-20261020

**Zakres:** kontrola techniczna odzyskanych produktów E0048; bez odczytu etykiet, rozstrzygnięć, wyników, predykcji ani ekonomiki.

## Wniosek

**Poprawność nieudowodniona dla kolejnego etapu oceny.** Odzyskane pliki są kompletne i spójne z terminalnym checkpointem w zakresie sprawdzonych hashy, schematów, kluczy, osi czasu i zapisanych snapshotów. Wspólna liczba 17 135 oznacza kompletność według źródłowej maski E0048, a nie historyczną gotowość predykcji: znacznik pochodzenia czasu odbioru cech jest UNKNOWN dla wszystkich 28 224 slotów. E0037 nadal ma niezaliczony T-60 parity gate, T0009 pozostaje zablokowane, a E0048 nie spełniło zamrożonego limitu 4 godzin.

Nie znaleziono dowodu na błąd wymagający podmiany odzyskanych produktów. Nie uzupełniano ani nie imputowano braków, nie zmieniano progu kwalifikacji i nie korygowano historycznych statusów. Produkty można używać do dalszego audytu źródeł w ściśle udokumentowanym zakresie; nie są dopuszczone do predykcji ani ewaluacji.

## Zakres, granice i stan repozytorium

- Development: przed 2026-07-01T00:00:00Z. Holdout: dokładnie [2026-07-01T00:00:00Z, 2026-10-07T00:00:00Z).
- E0048 v4: protokół SHA-256 ba79ec4d20787d69058609824514416338db5178a900cd4e9c26b3b84ac8f55a; zamrożony moduł cech SHA-256 00a9961d907b521edcfe7e15fefea079b7b47c82a36609a1b9f4b4924246ed4e; runner SHA-256 a9eb653d5d779a45523c56e8f8d59193874d41652b33132cd254a85ce5157fe3; finalizer SHA-256 57d7baded8f772c97b1ba1c205017723ab70bfd0bc93ca287905c0e4e1667542.
- Powiązane zamrożone identyfikatory: D0025, E0041 bundle SHA-256 7715419ce5efbce8405db861fb3a67c481038c07f11eec019137f14528f0f97d; E0047 checkpoint SHA-256 e3ed1ab15c2439ca4e9c3f82102ff1be97f23d2cb2a5cbe88ca7ff3b0980dae5; E0047 continuation manifest SHA-256 c6d73a6222e6cf5f95e6bb83c24d7a9810a3fe18c5a6ceba9834e845558928ec; E0043 gap inventory SHA-256 72f682b4a6be35baa03f6b1854ffdbe45f2cc4ca96fc2b267f82551296e0197c.
- Branch był zgodny z origin przed zmianami audytowymi, working tree był czysty, a po fetch HEAD i origin wskazywały 191766c. Nie wykonano resetu ani pełnego replayu.
- Skan procesów przy zamknięciu audytu nie wykazał działającego runnera, watchera ani finalizera E0048. Manifest nie zapisuje kodu wyjścia pierwotnego runnera ani finalizera; oba kody są **nieudokumentowane**. Sukces finalizacji potwierdzają zapisany manifest i hashe produktów, ale nie jest to dowód sukcesu pierwotnego runnera.
- Nie wywołano modelu ani ewaluatora. Nie odczytano kolumn celu, etykiet, rozstrzygnięć, payoutów ani wyników. Stan pozostaje evaluation_authorized=false, activated=false, HOLDOUT_NOT_RUN.

## Integralność produktów i zakresu

Terminalny checkpoint ma cursor 2 377/2 377; jego SHA-256 to b9f3a43fa66769e7cb3e01cb7b27b94f704d6806a0f484815c563440fc86903c, rozmiar 177 560 574 B. Liczniki checkpointu: 2 309 istniejących partycji, 420 112 552 wiersze projekcji i 416 545 575 odtworzonych zdarzeń. Zakres 2 377 godzin obejmuje 25 godzin rozgrzewki oraz 98 dni okna rynkowego, z godzinowymi punktami końcowymi zgodnymi z protokołem. W oczekiwanych godzinach brak dokładnie 68 partycji z przedziału [2026-08-15T10:00:00Z, 2026-08-18T06:00:00Z). Są to liczniki godzin/partycji i zdarzeń, nie liczba rynków.

W recovery finalizer sprawdził metadane 2 309 partycji i nie odczytał payloadu żadnego zdarzenia PMXT; nowego replayu podczas recovery nie rozpoczęto. Siedem plików produktów istnieje, jest czytelnych, ma zakończone zapisy i zgadza się z SHA-256 w manifeście. Łączny rozmiar tych produktów wynosi 157 340 584 B. Rekurencyjny katalog E0048 zajmuje 472 189 120 B w 73 plikach, w tym terminalny checkpoint i manifest audytu. Pomiar wolnego miejsca na D: 2026-11-03 około 08:39 UTC wyniósł 253 164 302 336 B.

| Produkt | Zakres/klucz | Wiersze i kolumny | Rozmiar B | SHA-256 |
|---|---|---:|---:|---|
| feature_rows.parquet | Opened, 2026-06-30 23:58–2026-10-06 23:53 UTC | 141 116 × 103 | 56 507 559 | abaa0360b2f7ae1d50c7befb1e62805cb684bbfb10d8f493c733ea65cb437b3a |
| holdout_input_rows.parquet | market_start_utc, 2026-07-01 00:00–2026-10-06 23:55 UTC | 28 224 × 231 | 48 903 134 | 4f64cf6664bfe61c999a2457410f8fa1e706f89b00fd851bc55ae409d7fbaab2 |
| slot_coverage.parquet | market_start_utc, ten sam kalendarz | 28 224 × 122 | 26 922 318 | a9bea5ab8aa960a6bad12cfb9be239b11474b765a48bdd382e2abc1cfa51c831 |
| t60_books.parquet | market_start_utc, snapshot T-60 | 28 224 × 51 | 12 492 531 | 6cb510d1ac7c710941f3a6c76400634b533d9690769e5ab965d6b3e3970b837d |
| t59_books.parquet | market_start_utc, snapshot T-59 | 28 224 × 51 | 12 513 679 | 2c999bfaddbd6c61d1c11e54ba0b364ec45f427670f2f667311f1428eb76f30f |
| regime_coverage_summary.csv | single, TWAP30, TWAP60 | 3 × 15 | 599 | 8b366e8b726f8b46a6b5e507cf0eb95bb89423e3f2e2d1b96656ea18c687d8ca |
| slot_coverage_summary.csv | techniczne miary pokrycia | 19 × 3 | 764 | 180ab8187296533520a98e4dec80305d1fc5041fee475cdeb151586b6e993ae3 |

Klucze kalendarza i cech są unikalne; 141 116 minut cech jest ciągłych na dokładnej siatce minutowej i zgodnych z oczekiwanym zakresem. Złączenia Gamma/PMXT wykonano wyłącznie po pięciu polach technicznych: condition_id, market_slug, market_start_utc, up_token_id i down_token_id. Niezależny rejoin dokładnie odtworzył zapisane rekordy: 80 127 wierszy Gamma i 50 184 PMXT w projekcjach źródłowych; 28 222 potwierdzone tożsamości, dwa sloty UNKNOWN, bez duplikatów niepustych identyfikatorów rynku/tokena. Dwa UNKNOWN nie są potwierdzeniem nieistnienia rynku.

## Uzgodnienie kalendarza i kwalifikacji źródłowej

| Etap | Liczba | Uwaga |
|---|---:|---|
| Sloty kalendarza | 28 224 | 5-minutowa siatka w zamrożonym półotwartym oknie |
| Potwierdzona tożsamość rynku/tokenów | 28 222 | dwa pozostałe sloty UNKNOWN |
| Obecny wiersz cech | 28 224 | pełne klucze minutowe; 28 223 mają komplet nie-NaN |
| Dokładne złączenie FLOW6 | 28 224 | join UTC bez przybliżania |
| Dwie natywne strony księgi poprawne na T-60 | 18 649 | bez komplementacji; bez mismatch BBO |
| Dwie natywne strony księgi poprawne na T-59 | 17 704 | bez komplementacji; bez mismatch BBO |
| Źródłowa maska E0048: source_only_components_complete | 17 135 | 28 224 minus 11 089 niekompletnych/quarantine |
| Opłata znana na obu cutoffach | 25 151 | nie oznacza kwalifikacji samodzielnie |

Regime counts zgadzają się z zamrożonym kalendarzem: single 10 656, TWAP30 2 016, TWAP60 15 552. Potwierdzone identyfikacje w tych reżimach wynoszą odpowiednio 10 655, 2 016 i 15 551; feature_ready odpowiednio 10 656, 2 016 i 15 551.

Przecięcie 17 135 niezależnie sprawdzono również przy wymaganiu potwierdzonej tożsamości, gotowych cech, dokładnego FLOW, poprawnych bezpośrednich ksiąg obu stron na obu cutoffach, metadanych wejścia/świeżości, znanej opłaty i braku przyszłego zdarzenia. To zgodność technicznych pól źródłowych. Runner definiuje source_only_components_complete wężej jako feature_ready × confirmed identity × source_quality_clean; nie obejmuje on historycznego czasu odbioru cech. Dla wszystkich 28 224 slotów source_ready_receive_time_provenance = UNKNOWN. Zatem 17 135 nie jest liczbą rynków z dowiedzioną historyczną gotowością predykcji.

Znany 68-godzinny brak wpływa na 1 115 potwierdzonych okien T-59: 816 bez dostępnego źródłowego entry-hour/freshness window oraz 299 z entry-hour, lecz niekompletną wcześniejszą historią. Te liczby uzgadniają zamrożony E0043/T0015 podzbiór, a nie dodatkową kategorię do dodania do 11 089. Dwa UNKNOWN identity pozostają osobno: starty 2026-07-12T02:25Z i 2026-09-03T15:20Z; nie są doliczane do 1 115.

Wartości NaN: dwa pola rp_short_resistance_center_dist w minutach Opened 2026-08-25 02:18 i 02:19 UTC; brak nieskończoności. Minuta 02:18 odpowiada slotowi rynku 02:20 i obniża jego feature_ready; minuta 02:19 nie mapuje się na start rynku 5-minutowego. 28 223/28 224 slotów ma feature_ready. Znacznik no_future_event_at_entry jest true dla 28 222 potwierdzonych slotów i null dla dwóch UNKNOWN. no_future_source_event_at_entry: T-60 28 218 true, 4 false, 2 null; T-59 28 219 true, 3 false, 2 null. Fałszywe flagi pozostają wyłączone przez kwalifikację: T-60 starts 2026-08-23 22:40, 2026-08-25 07:20, 2026-10-02 08:25 i 08:30; T-59 starts 2026-08-23 22:35 i 2026-10-02 08:25 oraz 08:30 UTC.

## Cechy, czasy i pochodzenie

Pełny odczyt technicznych produktów potwierdził kolejność 102 zamrożonych cech E0041: BASE70 (BASE64 + FLOW6), memory PCA8, ESN PCA8. Wszystkie dziewięć członków D zachowuje zamrożoną kolejność 86 wejść; testy zgodności członków w zapisanych artefaktach pasują. Typy cech: 61 float64, 3 int32 i 38 float32; FLOW6 jest float32. Wiersze input zawierają 231 kolumn technicznych, cech i ksiąg, bez kolumn target/label/outcome/resolution/payout/prediction/PnL/trade/EV. Feature values są równe odpowiadającym wierszom holdout_input_rows po dopasowaniu nazw kolumn i equal-NaN.

Mapowanie czasów jest zgodne z E0048: T-60 i moment decyzji = market_start − 60 s; moment Opened dla cech = market_start − 120 s; T-59 = market_start − 59 s. Potwierdzone snapshoty T-60 i T-59 mają entry time równe cutoffowi dla 28 222 rynków. Pierwszy slot holdoutu o 00:00Z korzysta z cechy Opened 23:58Z i kontynuacji stanu E0047 po ostatnim stanie 23:57Z; 23:58Z jest przetwarzane raz zgodnie z protokołem. Granice single/TWAP30/TWAP60 są zgodne z etykietą techniczną w protokole.

Na wszystkich kwalifikowanych wierszach ksiąg sprawdzono natywną tożsamość tokena UP/DOWN z metadanymi oraz flagi komplementacji, BBO, crossing i ambiguity. Dla każdej z 18 649 T-60 i 17 704 T-59 poprawnych par stron obie strony mają ask levels. Po prawidłowym zdekodowaniu list PyArrow wszystkie ceny ask są rosnące, w zakresie (0,1), rozmiary dodatnie, a pierwszy poziom zgadza się z zapisanym best ask i best ask size z tolerancją bezwzględną 1e-12. Liczba poziomów: UP 13–73, DOWN 12–75. Nie stwierdzono komplementowanych/nieznanych quote tokenów ani błędu porównywalnego BBO w tych kwalifikowanych wierszach.

Bundle E0041 i kolejności członków są przypięte hashami, a granica fitu przed pierwszą decyzją holdoutu jest 2026-06-30T23:59:00Z; zapis treningu wskazuje ostatnią dostępność etykiety 23:58Z, kalibracja ostatnie release 23:56:22Z. Ten audyt nie odczytywał ukrytych wyników i nie dowodzi całego pochodzenia selekcji/transformacji. W manifeście E0041 pole official_gamma_development_rows pozostaje NOT_VERIFIED_IN_THIS_AUDIT. Pełna historyczna gotowość cech też pozostaje UNKNOWN z powodu braku receive-time provenance. Nie należy traktować obecnych hashy plików jako historycznych fingerprintów, których wcześniej nie zapisano.

## T-60/T-59: zakres dowodu i niezależny replay próbki

Recovery finalizer odtworzył produkty z terminalnego checkpointu i porównał każdy T-60/T-59 wiersz: 49 pól skalarnych dla każdego cutoffu oraz wszystkie zagnieżdżone poziomy ceny/ilości. Porównanie zakończyło się zerową liczbą różnic bez tolerancji numerycznej po normalizacji wyłącznie reprezentacji kontenerów PyArrow arrays-of-arrays versus replay lists/tuples. To pełnozbiorowy dowód zgodności zapisanego produktu z tym samym terminalnym checkpointem, nie niezależny replay całego RAW.

Aby sprawdzić RAW i granice, przed odczytem event payloadów wybrano pięć technicznych startów: 2026-07-01 00:00Z (pierwszy slot holdoutu), 2026-08-07 00:00Z i 2026-08-14 00:00Z (granice reżimów), 2026-08-18 07:00Z (po luce), 2026-10-05 12:00Z (po granicy wznowienia). Limit: pięć rynków, ich wymagane 25-godzinne stany początkowe i snapshoty do T-59; zero pełnego odtworzenia historii. Zakres to 125 oczekiwanych slotów godzinowych, z których 101 miało plik, a 24 przypadały na znaną lukę; odczytano 19 335 429 wierszy projekcji i odtworzono 68 985 przefiltrowanych zdarzeń, używając wyłącznie 13 zamrożonych technicznych kolumn. Tymczasowy checkpoint w repozytorium usunięto po kontroli.

Dla wszystkich pięciu rynków obie migawki T-60 i T-59 zgadzają się z produktem dokładnie: pola skalarne i wszystkie pary poziomów ceny/ilości, zero tolerancji. Próbka nie zawierała naturalnego zdarzenia dokładnie na cutoffie. Kod implementuje stabilne sortowanie po czasie odbioru i pozycji wiersza Parquet, przetwarza pełną grupę odbioru, a następnie robi snapshot inclusive; wymaga exact market/token/side i zachowuje flagę zdarzenia ze źródłowym timestampem w przyszłości. Istniejący test syntetyczny sprawdza przed/równo/po cutoffie oraz grupy dzielone między batchami; w zapisanym prereplay logu E0048 miał wynik 2/2. Testów kodu nie uruchamiano ponownie w tym audycie.

Ta próbka nie dowodzi wszystkich 28 224 snapshotów względem niezależnego RAW. Nie zachował się wcześniejszy checkpoint pierwotnego runnera dokładnie z cursor 2 340, więc niezależne odtworzenie ciągłości stanu przez tę konkretną granicę resume pozostaje nieudowodnione. Należy rozdzielić tę lukę dowodową od pełnozbiorowego porównania finalizera.

## Wydajność i kody zakończenia

Protokół v4 ustala 14 400 s i limit RSS 24 GiB. Timer w runnerze startuje na wejściu _run_preparation przed wczytaniem protokołu i resource gate. Limit czasu jest sprawdzany po budowie cech, replayu ksiąg, materializacji produktów i reconciliation pokrycia, przed liczeniem hashy wyjściowych oraz raportem/manifestem. Nie wolno resetować zegara dla kontynuacji lub recovery.

Start: 2026-10-30 01:04:44Z. Najwcześniejszy udokumentowany moment ukończenia produktów to 2026-11-03 00:18:03.894578Z, co daje dolną granicę 342 799.9 s (95 h 13 min 19.9 s), czyli około 23.8× limitu 14 400 s. Czas kalendarzowy to dolna granica, nie aktywny czas CPU. Zapisane czasy aktywne: suffix replay 5 380.4 s dla 37 partycji; poprawiony terminal builder 43.8277 s; pełny aktywny czas oryginalnego runnera **nieznany**, przerwy nieudokumentowane, kod wyjścia runnera i finalizera nieudokumentowany. Peak RSS replayu: 3 546 705 920 B. Historyczny status pozostaje PARTIAL; manifest zachowuje SOURCE_PRODUCTS_RECOVERED_BUDGET_EXCEEDED / NOT_ACCEPTED_WITHIN_FROZEN_4_HOUR_BUDGET.

Profil z 2026-10-31 wskazał pandas group iteration/materialization jako główny koszt na jednej partycji. Kandydat uzyskał 50.05× na tej próbce i dokładną kontrolę stanu/ksiąg, ale nie został włączony, bo bezpieczne wznowienie z żywego checkpointu nie było dowiedzione. Uzasadniona poprawka na przyszłość: osobny zamrożony przebieg kandydata po sprawdzeniu restartu, identyczności wszystkich stanów/wierszy i zgodności końcowych snapshotów; następnie mierzony end-to-end budżet. Ten audyt nie benchmarkował ponownie ani nie zmieniał runnera.

## Werdykty

| Kategoria | Status | Dowód i granica |
|---|---|---|
| Integralność produktów | PASS w udokumentowanym zakresie | Siedem hashy, schematy, klucze, zakres i zapisane checkpointowe snapshoty sprawdzone; nie dowodzi niezależnego RAW dla całej historii. |
| Pokrycie danych | PARTIAL | 17 135 według source-only maski; 68 h/1 115 okien, dwa identity UNKNOWN, jedna niegotowa cecha i liczne jawne flagi pozostają. |
| Przyczynowość/pochodzenie | PARTIAL / UNKNOWN | Granice i kod cutoffu sprawdzone; pięć replayów RAW zgodnych. Historyczne receive-time dla cech UNKNOWN dla wszystkich slotów, E0037 parity nadal FAILED, pochodzenie official_gamma_development_rows NOT_VERIFIED_IN_THIS_AUDIT i resume na cursor 2 340 bez niezależnego dowodu. |
| Wydajność | FAIL wobec zamrożonej bramki | Dolna granica 342 799.9 s wobec 14 400 s. Aktywny czas pełnego runu nieznany. |

## Następny krok — plan, bez wykonania

Nie uruchamiać ewaluacji teraz. Następny etap powinien najpierw przygotować osobny, zamrożony plan zamknięcia bramek T0009/E0037 oraz wydajności E0048: rozstrzygnąć możliwe do dowiedzenia pochodzenie czasu gotowości, udokumentować pozostałe nierozstrzygalne statusy bez imputacji, naprawić/przypiąć źródłowy dowód T-60 parity, oraz zmierzyć kandydata wydajności wraz z odtworzeniem stanu po wznowieniu i pełną równoważnością produktów. Nie resetować ani nie przepisywać statusu E0048; odrębne wykonanie wymaga nowego protokołu i nie może zmienić semantyki E0048.

Dopiero po zamknięciu tych bramek można rozważyć jednorazową ocenę według E0041 D bundle (dziewięć D members, stały ensemble i kalibracja) oraz zamrożonego E0041/Evaluator: run_btc_D_holdout_evaluation_20261028.py i configs/research/btc_D_holdout_preparation_20261028_v2.json. Protokół przewiduje dokładny holdout, bez refitu ani aktualizacji kalibracji, payout delays 1/5/15 s i kontrole fixed_5_usd oraz free_cash_5pct. Wymaga to osobnego, jawnego dopuszczenia ewaluacji; obecny stan evaluation_authorized=false, activated=false, HOLDOUT_NOT_RUN pozostaje w mocy. Historyczna ekspozycja holdoutu oznacza, że byłaby to wcześniej przewidziana stała ocena, nie virgin ani niezależny test.

## Artefakty i rejestry

- [Maszynowy manifest audytu](../../data/analysis/polymarket/BTC/btc_D_historical_inputs_20261030_v4/recovered_inputs_audit_20261103.json)
- [Oryginalny raport recovery](btc_D_historical_input_preparation_20261030_recovery.md)
- [Oryginalny manifest recovery](../../data/analysis/polymarket/BTC/btc_D_historical_inputs_20261030_v4/input_preparation_recovery_manifest.json)
- [Protokół E0048 v4](../../configs/research/btc_D_historical_input_preparation_20261030_v4.json)
- [Status projektu](../../PROJECT_STATE.md), [TASKS](../../TASKS.md), [EXPERIMENTS](../../EXPERIMENTS.md) i [historia BTC](../../docs/BTC_RESEARCH_STATUS_20261010.md)

Nie zmieniono kodu, więc testów nie uruchamiano. Walidacja obejmuje kontrole tylko do odczytu i końcowe git diff --check.
