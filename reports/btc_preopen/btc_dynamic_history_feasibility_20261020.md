# BTC dynamic-history feasibility — 2026-10-20

## Rozstrzygnięcie

`original_gate = FAIL`: **0 kwalifikowanych obserwacji** po wymaganej kontroli dostępności RAW. Wynik pozostaje bez zmian. Nie uruchomiono treningu, tuningu ani oceny ekonomicznej; nie odczytano ani nie wykorzystano wierszy holdoutu; nie pobrano pełnego archiwum.

Oddzielny licznik zapisanej kwalifikacji książki/H, z pominięciem wyłącznie bramki timestampu RAW, wynosi **637 rynków ewaluacji** (520 w foldzie 1 i 117 w foldzie 2). Niezależny replay sprawdził w ewaluacji 44 kwalifikujące się przypadki: 40 pierwotnych i 4 odzyskane; nie był to pełny replay wszystkich 637. To pokrycie pomocnicze, nie zaliczenie gate’u. Historyczne `637` ani `908` nie są wynikiem poprawionej bramki.

Nie ma obecnie udokumentowanej ścieżki do wymaganej liczebności przy zachowaniu zamrożonych cech i reguł. Dane książki mogą uzupełnić część luk tylko po ponownym sprawdzeniu każdego rynku; żadne dostępne źródło nie dostarcza brakującego dowodu przyczynowej dostępności RAW. Z tego powodu wykonalność tego eksperymentu historycznego pozostaje **zablokowana**.

## Zakres zamrożony i metoda

- Ewaluacja: `[2026-05-15T00:00:00Z, 2026-07-01T00:00:00Z)`.
- Fold 1: ewaluacja `[2026-05-15, 2026-06-01)`, pierwsza decyzja `2026-05-14T23:59:00Z`.
- Fold 2: ewaluacja `[2026-06-01, 2026-07-01)`, pierwsza decyzja `2026-05-31T23:59:00Z`.
- Trening pozostaje rozszerzającym się prefiksem; etykieta musi być oficjalnie opublikowana ściśle przed pierwszą decyzją foldu. Nie przesunięto granic, warm-upu, foldów, progów, H ani reguł kwalifikacji.
- Każdy odczyt z Parquet obejmował filtr development `market_start_utc < 2026-07-01T00:00:00Z` przed materializacją. Dla prefiksów treningowych zastosowano dodatkowe ograniczenie do ich zamrożonej decyzji. Nie materializowano post-cutoff.

## Rozłączne etapy pokrycia

„Oficjalny rynek” oznacza w tej tabeli wiersz z indeksu rynku użytego i zahashowanego przez zamrożoną konfigurację. `RAW match` to dokładne złączenie po czasie startu; wszystkie dopasowane wiersze mają brak timestampu dostępności. `T−60 row` oznacza obecność rekordu snapshotu, nie jego poprawność. „Kandydat H” to obecność w zamrożonej puli 6 796 statycznych wejść. Przyczyny po tej kolumnie są **rozłączne — pierwsza niespełniona kontrola statycznego wejścia**. H oraz q=$5 są pokazane po tej selekcji.

| Zakres | Oficjalne starty | RAW match / brak | Wiersz T−60 | Kandydat H | Rozłączne odrzucenia poza kandydatem H | H poprawne bez RAW | Etykiety treningowe opublikowane przed decyzją | q=$5 znane po obu stronach | RAW on-time — ścisły gate |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| Trening foldu 1: starty przed `2026-05-14 23:59Z` | 8 433 | 8 433 / 0 | 8 433 | 1 927 | 4 351 źródło/native book; 2 155 niejednoznaczny ask | 1 760 | 1 760, obie klasy | 908 | **0** |
| Ewaluacja foldu 1 | 4 895 | 4 895 / 0 | 4 895 | 2 144 | 1 605 źródło/native book; 1 146 niejednoznaczny ask | 520 | — | — | **0** |
| Trening foldu 2: starty przed `2026-05-31 23:59Z` | 13 328 | 13 328 / 0 | 13 328 | 4 071 | 5 956 źródło/native book; 3 301 niejednoznaczny ask | 2 280 przed filtrem release; 2 279 po nim | 2 279, obie klasy | 1 207 | **0** |
| Ewaluacja foldu 2 | 8 634 | 8 633 / 1 brak RAW | 8 634 | 2 725 | 5 202 źródło/native book; 485 niejednoznaczny ask; 214 niekompletne okno archiwum; 7 luka po inicjalizacji; 1 brak RAW | 117 (113 pierwotnie + 4 niezależnie odzyskane) | — | — | **0** |

Indeks zawiera **13 529** startów w całej zamrożonej ewaluacji. W 6 796 kandydatów audytu wchodzą `1 927 + 2 144 + 2 725`; pozostałe 15 166 z 21 962 lokalnych development startów nie weszły do audytu H. W samym okresie ewaluacji 8 660 startów nie weszło do puli 4 869 kandydatów.

Wśród 6 796 kandydatów wszystkie 6 796 mają kompletną lokalną ciągłość źródła dla audytowanych okien, zapisane 4/4 endpointy (`T−120`, `T−75`, `T−65`, `T−60`) i inicjalizację przed `T−120`; braków endpointów ani niekompletnych okien nie stwierdzono w tej **wyselekcjonowanej** puli. Zamrożone 21 cech H przeszło 2 393 rynków w zachowanej tabeli stanu. Niezależny replay odtworzył 1 469 rynków łącznie: 60 wcześniej zaakceptowanych próbek i 4 odzyskane przypadki. W ewaluacji niezależnie sprawdzono 40 pierwotnie zaakceptowanych i 4 odzyskane. Pełna pula kwalifikacji książki/H przed RAW nadal wynosi 637.

W 21 962 lokalnych development startach rekord T−60 istnieje dla każdego rynku. To nie oznacza poprawnej książki: na całym development `quote_valid` jest fałszywe dla 11 122 rekordów, a `has_full_snapshot` dla 5 366. W foldzie 2 ewaluacji 335 rekordów ma brakujące albo nieprzetworzone godziny okna, 45 ma flagę luki po inicjalizacji, a 36 nie ma dostępnego/fresh okna źródła. Te flagi nakładają się na siebie i na odrzucenia statyczne.

## Dlaczego audyt ma 6 796, a indeks 21 962

Audyt H nie wybierał reprezentatywnej próbki z indeksu. Jego wejściem były `static_inputs.parquet` i `dynamic_inputs.parquet`, a kod audytu wymaga dokładnie 6 796 statycznych kandydatów i buduje endpointy tylko dla tych ID. Statyczny etap wcześniej wymagał dopasowanego RAW, wiersza T−60 oraz poprawnych źródeł, pełnej natywnej książki, drabin i $5 głębokości. Historyczny runner przepuszczał w tym miejscu `NaT`, stąd te obiekty zachowały się mimo pustego dowodu dostępności.

Rozbieżność `21 962 − 6 796 = 15 166` nie jest równoznaczna z 15 166 rynkami odzyskiwalnymi ani z brakiem 15 166 oficjalnych rynków. Każdy z nich ma wiersz indeksu i T−60, lecz nie przeszedł wejściowych kontroli książki/źródła albo selektora. Najczęstsze **pierwsze** odrzucenia w czterech zakresach są w tabeli powyżej; w foldzie 2 ewaluacji łączą się dodatkowo kontrole brakującego RAW, okna archiwum i luki po inicjalizacji. Przykładowe flagi nakładające się na całą oficjalną populację ewaluacji:

| Flaga snapshotu (liczona jako unia UP/DOWN) | Fold 1 ewaluacji (4 895) | Fold 2 ewaluacji (8 634) |
|---|---:|---:|
| Niepoprawny quote | 1 605 | 5 166 |
| Brak full snapshotu | 318 | 5 046 |
| Niejednoznaczna kolejność ask | 2 043 | 607 |
| Książka crossed | 1 287 | 120 |
| Brak entry/freshness okna archiwum | 0 | 36 |
| Luka po inicjalizacji UP albo DOWN | 0 | 45 |
| Brakujące lub nieprzetworzone godziny okna | 0 | 335 |

Flagi są diagnostyczne i nakładają się; nie należy ich sumować. Z kolei rozłączne przyczyny odrzucenia statycznego są liczone tylko raz na rynek.

## H: rozłączne odrzucenia i nakładające się flagi

Wśród kandydatów audytu pierwsza przyczyna H rozkłada się następująco:

| Zakres | Kandydaci | H zaliczone przed niezależną korektą | Odzyskane niezależnie | Nadal odrzucone: unresolved / history | reset w oknie H60 | invalid quote H60 | invalid quote H15 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Trening foldu 1 | 1 927 | 1 760 | 0 | 58 | 64 | 44 | 1 |
| Ewaluacja foldu 1 | 2 144 | 520 | 0 | 1 179 | 177 | 268 | 0 |
| Prefiks treningu foldu 2 przed filtrem release | 4 071 | 2 280 | 0 | 1 237 | 241 | 312 | 1 |
| Ewaluacja foldu 2 | 2 725 | 113 | 4 | 1 514 | 751 | 347 | 0 |

Uwaga: w ostatnim wierszu cztery odzyskane rynki pochodziły z pierwotnej klasy resetu; po ich niezależnej weryfikacji pozostaje 2 608 odrzuceń H. Progi oraz gate nie zostały zmienione.

Poniższe flagi są liczone jako unia po trzech horyzontach H i nakładają się. To liczniki surowego audytu flag, więc wśród resetów folda 2 pozostają także cztery później zweryfikowane identyczne snapshoty.

| Flaga H | Trening F1 | Ewaluacja F1 | Ewaluacja F2 |
|---|---:|---:|---:|
| `book_message_counted_as_reset` | 133 | 1 618 | 2 606 |
| `equal_timestamp_or_ambiguous_event` | 0 | 1 395 | 1 693 |
| `equal_timestamp_price_conflict` | 0 | 1 389 | 1 595 |
| `invalid_native_quote` | 45 | 356 | 572 |
| `ignored_same_side_stale_price_change` | 71 | 22 | 153 |
| `ignored_stale_book` | 9 | 20 | 89 |
| `accepted_cross_side_late_price_change` | 22 | 3 | 13 |

## q=$5 w treningu

Pomocnicze, book-only liczby q=$5 po obu stronach to 908 w treningu foldu 1 i 1 207 w treningu foldu 2; próg wynosi 1 000 w każdym foldzie. Fold 1 jest 92 rynki poniżej progu, fold 2 przekracza go o 207. Wszystkie te rekordy mają oba `depth_known`, status q5 `settled` po UP i DOWN oraz `t59_source=pmxt_v2`. Po zastosowaniu wymogu RAW on-time użyteczne q5 wynosi **0/0**, bo brak timestampu dostępności RAW dotyczy każdej z tych obserwacji.

## Preflight braków i osiągalność źródeł

Preflight obejmował trzy konkretne rynki development z rozłącznych przyczyn odrzucenia. Czytano tylko potrzebne lokalne partycje godzinowe i okna do decyzji; nie pobierano pełnego archiwum.

| Przykład | Zakres / start | Wynik lokalnych zdarzeń dla UP/DOWN | Wniosek dla zamrożonych reguł |
|---|---|---|---|
| Brak statycznego wejścia z niepoprawnym źródłem/książką | Trening F1, `2026-05-08T08:55Z` | 3 432 zdarzenia w 15-minutowym lookbacku; 288 w oknie `[T−120,T−60]`, ale żaden nie jest snapshotem `book`, a w 15 min nie ma snapshotu `book` dla obu tokenów. T−60 oznacza invalid quote, niejednoznaczny UP ask i crossed książki. | Ponowne pobranie tych samych zmian `price_change` nie odtworzy wymaganego punktu inicjalizacji ani poprawnego quote. Przykład nie jest odzyskiwalnym rynkiem H. |
| Brak statycznego wejścia z niejednoznacznym ask | Ewaluacja F1, `2026-05-22T21:55Z` | 1 356 zdarzeń w 15 min; 475 w wymaganym oknie; są snapshoty inicjalizujące oba tokeny przed T−120. T−60 ma `down_ask_order_ambiguous=true`. | Źródło istnieje, ale dane tej próbki nie spełniają niezmienionej kontroli poprawności książki. |
| Niekompletne okno źródła | Ewaluacja F2, `2026-06-11T04:05Z` | Lokalna partycja `2026-06-11T04` nie istnieje; dla ID rynku nie znaleziono zdarzeń w sprawdzonych 15 min ani w wymaganym oknie. Jednorazowy HTTP HEAD do obiektu PMXT v2 `2026-06-11T04` zwrócił 404. | Sprawdzone źródło nie uzupełnia tej luki. Nie wolno uznać brakującej godziny za pełną historię dla tego rynku. |

Dokumentacja PMXT v2 opisuje publiczne godzinowe pliki z `book` i `price_change`, start pokrycia `2026-04-13T19Z`, deklarowane pełne pokrycie live marketów i bezpośredni URL; dokumentacja mówi też, że brakujący klucz oznacza godzinę bez zdarzeń. To jest droga do sprawdzania konkretnych eventów od tej daty, ale nie dowód, że każdy z brakujących kandydatów spełni gate: lokalna kopia ma 1 853 z 1 856 godzin development, a badany plik z 11 czerwca zwrócił 404. Pliki 100–400 MB/godzinę w globalnym archiwum nie były pobierane. [Dokumentacja PMXT v2](https://archive.pmxt.dev/docs/v2-data-overview)

Jako alternatywny trop strona PredictionTicks pokazuje 288 rynków BTC z pełnymi drabinami w dniu 13 czerwca i około 1,5-sekundowy odstęp próbek; pełne archiwum jest oferowane za $15. Takie snapshoty mogą pomóc sprawdzić stan T−60/T−59, ale nie zawierają natywnego strumienia `book`/`price_change` z dokładną kolejnością i receive/source timestamps potrzebnymi do niezmienionego H i depletion. Nie pobrano ani nie kupiono tych danych i nie zaliczono ich do pokrycia. [Przykładowy dzień PredictionTicks](https://predictionticks.com/datasets/btc-orderbook-5m/2026-06-13)

Wewnątrz zamrożonej ewaluacji nie brakuje startów w lokalnym indeksie: jest ich 13 529. „Brakujące rynki ewaluacji” w obecnym wyniku to przede wszystkim rynki spoza puli statycznej albo odrzucone przez H, a nie brakujące wiersze indeksu. Dla treningu F1 lokalny indeks ma 8 433 starty od `2026-04-15T17:05Z`; pliki eventów zaczynają się 25 godzin wcześniej, ale w tych wcześniejszych godzinach nie ma zweryfikowanego wspólnego indeksu rynku i RAW. Starsze RAW OOF bez joinu do oficjalnego rynku oraz bez timestampu dostępności nie zwiększa liczebności treningu. Dane sprzed 15 kwietnia nie rozszerzają ewaluacji.

## Dostępność i przyczynowość RAW

- Lokalny RAW OOF ma 318 560 skończonych predykcji; `prediction_available_at_utc` jest brakujący w 318 560/318 560. Z 21 962 lokalnych development startów 21 961 ma RAW po joinie; wszystkie 21 961 mają dostępność nieudowodnioną, zero jest wykazanych jako spóźnione, a jeden start `2026-06-30T23:55Z` nie ma RAW.
- Znany jednorazowy przegląd lokalizacji manifestu OOF i logów jego przebiegu nie wykazał zarejestrowanego czasu odbioru wejść ani gotowości predykcji. Nie powtarzano poszukiwań; nie pojawiła się nowa lokalizacja ani nowe źródło logów.
- Zapisane modele mogą odtworzyć liczby RAW, lecz nie dowodzą czasu dostępności ani przyczynowości całego pipeline’u. Wcześniejsze OOF zależą od selekcji cech i tuningu wykonanych na pełnym development, bez fold-local selekcji/tuningu. Nie dopisano czasów dostępności z nominalnego timestampu świecy.
- Brakującego dowodu nie naprawia uzupełnienie książki. Aby uzyskać poprawne historyczne predykcje, potrzebny byłby nowy, fold-local pipeline na development oraz pomiary/wiarygodne logi dostępności wejść i zakończenia inferencji. Samo odtworzenie ze starych modeli pozostawia wcześniejsze predykcje zależne od przyszłego wyboru cech/parametrów.

## Gate, przedziały i rekomendacja

| Kryterium zamrożonego gate’u | Wynik ścisły | Próg | Status |
|---|---:|---:|---|
| Kwalifikowane rynki ewaluacji | **0** | 2 000 | FAIL |
| Dni z kwalifikowanymi obserwacjami | **0** | 20 | FAIL |
| Niepuste foldy ewaluacji | **0** | 2 | FAIL |
| Oficjalne etykiety release przed decyzją, F1 / F2 | **0 / 0** z RAW on-time | 1 000 w każdym niepustym foldzie i obie klasy | FAIL |
| Kwalifikowane q=$5 po obu stronach w treningu F1 / F2 | **0 / 0** z RAW on-time | 1 000 w każdym niepustym foldzie | FAIL |

Liczebność rynków:

- **Zapisany pass H/książki przy pominięciu RAW:** 637 ewaluacyjnych; z tego niezależnie odtworzono 44 (40 pierwotnie zaliczonych próbek i 4 odzyskane). Pozostałe nie są pełnym niezależnym replayem.
- **Dodatkowe rynki udowodnione jako odzyskiwalne w tym preflight:** 0. Cztery niezależnie odzyskane wcześniej rynki są już zawarte w 637.
- **Nieweryfikowana pula book-only:** 12 892 oficjalne starty ewaluacji — 4 232 z kandydatów 6 796 odrzucone przez H oraz 8 660 poza selektorem statycznym. To górna granica puli do sprawdzenia, nie prognoza odzysku.
- **Absolutny limit book-only w lokalnym indeksie:** 13 529 startów; spełnienie 2 000 wymagałoby jeszcze co najmniej 1 363 kwalifikujących się rynków poza obecnym wynikiem 637.
- **Limit ścisłego gate’u przy obecnych dowodach RAW:** 0. Nie ma udokumentowanego historycznego czasu RAW on-time dla żadnego rynku.

**Rekomendacja B — zamknąć wykonalność obecnego eksperymentu jako zablokowaną.** Nie uruchamiać dalszego pełnego replayu ani refitów, dopóki nie ma (1) wiarygodnych logów czasu odbioru danych wejściowych i gotowości RAW albo jawnej zgody na odtworzenie nowego fold-local pipeline’u, (2) kompletnego, dokładnego strumienia zdarzeń książki i snapshotów dla brakujących rynków w `[2026-05-15, 2026-07-01)` oraz treningu pierwszego foldu, i (3) źródłowo potwierdzonych T−60/T−59 drabin z danymi q=$5. Obecny lokalny indeks wystarcza jako populacja startów, lecz nie jako dowód spełnienia H.

Nie wykonano refitu. Szacunek obliczeniowy z poprzedniego audytu — nie benchmark — wynosił około 5–8 h dla jednej selekcji prefiksowej i dwóch fitów foldowych albo 15–24 h przy powtarzaniu selekcji przy trzech originach. To koszt przyszłej przyczynowej ścieżki, nie naprawa historycznych timestampów RAW. Nie oszacowano kosztu pozyskania brakujących dokładnych danych książki; dostępny preflight nie potwierdził kompletności takiego źródła.
