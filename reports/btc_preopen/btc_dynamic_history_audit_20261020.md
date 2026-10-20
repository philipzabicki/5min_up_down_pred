# Audyt historii dynamicznej BTC — 2026-10-20

Audyt: `btc_dynamic_history_audit_20261020_v1`. Kod bazowy: `3d95f641288f57e5f7d4c9ed77c621b5f678daaf`. Identyfikator eksperymentu: `btc_market_dynamics_action_value_20261020`.

## Wniosek

Potwierdzono błąd licznika `book_reset_count`: zwiększał się przed decyzją replayu dla każdej wiadomości `book`, także dla przyjętych, pełnostanowo identycznych snapshotów. W wąskiej grupie 992 rynków odrzuconych z tego powodu dokładny ślad wykazał 4 rynki z samymi przyjętymi identycznymi snapshotami. Po poprawce licznika i niezależnej kontroli stanu oraz cech odzyskano te 4 rynki; pozostałe 988 zawierały zaakceptowane snapshoty zmieniające pełny native book i nadal są odrzucone. Nie zmieniano cross-side, stale, ambiguity, quote validity, inicjalizacji, mianowników, granic czasu ani definicji 21 cech H.

Feasibility przeliczono przed fit: **637/2 000 rynków oceny; gate FAIL**. Obserwowane dni i niepuste foldy przechodzą, udostępnione etykiety przechodzą w obu foldach; fold 1 nadal ma 908/1 000 kwalifikowanych rynków q=$5. Wniosek: potwierdzony błąd poprawiono i feasibility przeliczono; oryginalny eksperyment pozostaje zablokowany. Rentowność nie była oceniana.

## Zakres, populacja i odtwarzalność

Audyt odtworzył pełną populację **6 796** statycznych kandydatów development z `2026-07-01T00:00:00+00:00` jako cutoffem wyłącznym. Wczytano wyłącznie rynki development, kończące się przed cutoffem. Oryginalne pliki, checkpointy i raporty pozostały nietknięte; wyniki zapisano pod `data/analysis/polymarket/BTC/btc_dynamic_history_audit_20261020_v1/`.

Weryfikacja artefaktów: statyczne/dynamiczne/endpointy/common/execution-targets zachowują hashe zamrożonego przebiegu; wszystkie cztery checkpointy workerów i ich wyjścia zgadzają się z manifestami. Hash pliku cech w lokalnym checkoutcie odpowiada wersjonowanemu manifestowi raportu (`63c171c4a3a4c8cfe818761f60033e815f43c4298368fdf93d0b5709469c7dc6`), ale nie hashowi wpisanemu do `frozen_protocol.json` (`f7d6d74e241dbe3718e1a4cb0e9cc50407f7f4d5b829f7ebe124a4409c363d82`). Nie znaleziono wyjaśnienia tej rozbieżności; zachowano ją jako ograniczenie pochodzenia. Kod runnera z bazowego commitu odpowiada zamrożonemu SHA; audytowa poprawka ma osobny hash `da18b6fdaa3358ab6c7ec815c2f01405340695689f29526b3d768d4b1867e5c0`. Konfiguracja nie zmieniła się.

Wersje raportu/konfiguracji mają hashe: raport `6ae314fd7461d5b5e41f8342b45a1133113928f4d8c185d747a1c8be5c1b09ca`, manifest `247b3128cf5fa155a84932dcf78ca48639482b12fb7a8ba63c8806bd3f68d111`, konfiguracja `f795b7d4a7564a691034a245f8ac3adf762bea349ef5c5e69f4141886856b6df`. Pełny crosswalk hashy i checkpointów: `data/analysis/polymarket/BTC/btc_dynamic_history_audit_20261020_v1/artifact_consistency.json`.

## Liczebności na tej samej populacji

Kategorie pierwszego odrzucenia są rozłączne i zależne od kolejności kontroli; flagi z kolejnych horyzontów nakładają się. Poniższa tabela zachowuje historyczne liczby bazowe i pokazuje wyłącznie potwierdzone przesunięcie czterech rynków.

| Okres UTC | Kandydaci | Dni | Przyjęte bazowo | Odrzucone bazowo | Przyjęte po audycie | Odrzucone po audycie | Pierwszy powód: unresolved | book reset bazowo → po | Quote 60 s | Quote 15 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Przed 15 maja | 1 927 | 30 | 1 760 | 167 | 1 760 | 167 | 58 | 64 → 64 | 44 | 1 |
| Fold 1 — 15–31 maja | 2 144 | 17 | 520 | 1 624 | 520 | 1 624 | 1 179 | 177 → 177 | 268 | 0 |
| Fold 2 — czerwiec | 2 725 | 13 | 113 | 2 612 | 117 | 2 608 | 1 514 | 751 → 747 | 347 | 0 |

| Pierwszy powód / status | Bazowo | Po audycie |
| --- | --- | --- |
| Przyjęte | 2 393 | 2 397 |
| unresolved_order_or_history_60s | 2 751 | 2 751 |
| book_reset_inside_depletion_window_60s | 992 | 988 |
| invalid_native_quote_window_60s | 659 | 659 |
| invalid_native_quote_window_15s | 1 | 1 |

Zmiany statusu odrzucony → przyjęty:

| Market ID | Start UTC | Pierwotna klasa | Po audycie |
| --- | --- | --- | --- |
| 0x0ef38ea42e4eca758ba2ba0cea70077e33497b65edf8fa3779aa7b07621a3046 | 2026-06-10T21:10:00+00:00 | accepted_identical_full_book | accepted — `book_reset_count` nie przyrasta dla identycznego snapshotu |
| 0x2a0bd277ad3d8b19f33deb28c1c38ac98870332c4cc4aab249d269151f9a7537 | 2026-06-11T02:50:00+00:00 | accepted_identical_full_book | accepted — `book_reset_count` nie przyrasta dla identycznego snapshotu |
| 0x93cdc459ec2b72b631a6501ddb4d76a80424732a2bd523d311af2252208739ed | 2026-06-10T18:10:00+00:00 | accepted_identical_full_book | accepted — `book_reset_count` nie przyrasta dla identycznego snapshotu |
| 0xae7e74be3ec0782e9607b934f062acff8772ef2f1fffb385e986a311cd7c96ba | 2026-06-05T23:40:00+00:00 | accepted_identical_full_book | accepted — `book_reset_count` nie przyrasta dla identycznego snapshotu |

Nie zmienił się żaden status przyjęty → odrzucony ani żaden niezwiązany powód. 4 zmiany odrzucony → przyjęty pochodzą wyłącznie z okresu 1–30 czerwca; 988 rynków pozostaje odrzuconych z powodu rzeczywistej zmiany pełnego book. Cztery rynki mają udokumentowane identyczne snapshoty, kompletną ciągłość źródła, przechodzą pozostałe flagi i mają skończone 21 cech H. Hipoteza wspólnego usunięcia obu gate’ów dawała 994 kandydatów; 2 dodatkowe rynki wymagają też pominięcia cross-side gate’u i nie zostały dopuszczone.

### Pierwszy powód a nakładające się flagi (cały okres)

Każda kolumna liczy rynki z flagą gdziekolwiek w horyzontach 5/15/60 s, również gdy pierwszym powodem był inny warunek.

| Pierwszy powód | Cross-side late | Wiadomość book jako reset | Ambiguity / equal ts | Konflikt equal ts | Stary price change | Stary book | Niepoprawny quote | Mianownik UP ≤0 | Mianownik DOWN ≤0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Przyjęty | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| unresolved_order_or_history_60s | 34 | 2 717 | 2 599 | 2 504 | 216 | 102 | 313 | 0 | 0 |
| book_reset_inside_depletion_window_60s | 0 | 992 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| invalid_native_quote_window_60s | 4 | 648 | 489 | 480 | 30 | 16 | 659 | 0 | 0 |
| invalid_native_quote_window_15s | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |

W horyzoncie 60 s: 38 rynków miało zaakceptowane cross-side-late, 246 miało zignorowane stale price changes, 118 zignorowane stale books, 3 088 ambiguity/equal-time flags i 2 984 konflikty timestampów. Cross-side late pozostaje zastosowane do własnej strony zgodnie z replayem; samodzielne usunięcie gate’u odzyskuje 0 rynków.
Inicjalizacja przed T−120 brakowała w 0 rynkach; ciągłość źródła jest kompletna dla 6 796/6 796. Liczba nie-dodatnich mianowników depletion według horyzontu/strony: `{"15s": {"down": 0, "up": 0}, "5s": {"down": 0, "up": 0}, "60s": {"down": 0, "up": 0}}`.

### Niezależne flagi według okresu i horyzontu

Wartości nie sumują się do liczby unikalnych rynków; pojedynczy rynek może mieć kilka flag. Pełna tabela po rynku zawiera czasy granic, liczniki start/end/przyrost, wszystkie flagi i powód statusu.

### 5 s

| Okres | Cross-side late | Wiadomość book jako reset | Ambiguity / equal ts | Konflikt equal ts | Stary price change | Stary book | Niepoprawny quote | Mianownik UP ≤0 | Mianownik DOWN ≤0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Przed 15 maja | 1 | 9 | 0 | 0 | 23 | 0 | 5 | 0 | 0 |
| Fold 1 — 15–31 maja | 0 | 994 | 52 | 42 | 2 | 1 | 43 | 0 | 0 |
| Fold 2 — czerwiec | 0 | 1 655 | 69 | 42 | 39 | 13 | 121 | 0 | 0 |

### 15 s

| Okres | Cross-side late | Wiadomość book jako reset | Ambiguity / equal ts | Konflikt equal ts | Stary price change | Stary book | Niepoprawny quote | Mianownik UP ≤0 | Mianownik DOWN ≤0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Przed 15 maja | 8 | 32 | 0 | 0 | 39 | 2 | 9 | 0 | 0 |
| Fold 1 — 15–31 maja | 0 | 1 393 | 156 | 141 | 4 | 4 | 83 | 0 | 0 |
| Fold 2 — czerwiec | 2 | 2 229 | 260 | 178 | 58 | 36 | 217 | 0 | 0 |

### 60 s

| Okres | Cross-side late | Wiadomość book jako reset | Ambiguity / equal ts | Konflikt equal ts | Stary price change | Stary book | Niepoprawny quote | Mianownik UP ≤0 | Mianownik DOWN ≤0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Przed 15 maja | 22 | 133 | 0 | 0 | 71 | 9 | 44 | 0 | 0 |
| Fold 1 — 15–31 maja | 3 | 1 618 | 1 395 | 1 389 | 22 | 20 | 268 | 0 | 0 |
| Fold 2 — czerwiec | 13 | 2 606 | 1 693 | 1 595 | 153 | 89 | 347 | 0 | 0 |

Pełna tabela audytowa: `data/analysis/polymarket/BTC/btc_dynamic_history_audit_20261020_v1/market_horizon_audit.parquet` — 6 796 wierszy po rynku; 149 kolumn, zawiera wszystkie trzy horyzonty i cztery endpointy. Schemat i SHA-256 w manifeście. Podsumowanie liczebności: `data/analysis/polymarket/BTC/btc_dynamic_history_audit_20261020_v1/endpoint_summary.json`.

## Semantyka zdarzeń i poprawka

| Przypadek | Zdarzenie wejściowe i zachowanie replayu | Licznik po poprawce | Decyzja gate’u |
| --- | --- | --- | --- |
| Pierwszy snapshot / reinitializacja | replay akceptuje i inicjalizuje native state | +1 | nadal blokuje okno, gdy wystąpi w oknie cechy |
| Snapshot zaakceptowany, identyczne bidy i aski na wszystkich poziomach | bez zmiany book; pełny stan porównywany po normalizacji cen | +0 | nie odrzuca samą wiadomością book |
| Snapshot zaakceptowany, zmieniony pełny book | stan przechodzi na snapshot; ścieżka depletion pozostaje nieznana | +1 | okno nadal odrzucone |
| Snapshot stary względem najnowszego native side | replay ignoruje, zwiększa out-of-order book | +0 | nadal odrzuca przez out-of-order |
| Konflikt equal timestamp / receive group | nadal zwiększa ambiguity i blokuje okno | tylko gdy zaakceptowana zmiana pełnego stanu | nadal odrzucone przez ambiguity/conflict |

Kontrprzykład syntetyczny: Dwa przebiegi zakończyły się identyczną pełną książką T−60, ale depletion względne wyniosło 0.5 vs 0; kontrola wykryła różnicę.

W 992 kandydatach książkowych prześledzono 15 382 wiadomości `book` w wymaganych oknach; wszystkie były przyjęte, bez stale snapshotów i konfliktów grupy. Cztery odzyskane rynki obejmowały łącznie 6 identycznych snapshotów. Kategoria 988 state-changing rynków pozostaje wykluczona; nie rekonstruowano ścieżki przez odejmowanie snapshotów.

### Niezależna kontrola wartości i stanu

- 5 876 endpoint rows z 1 469 rynków: 0 mismatch dla dokładnych liczników/flag, czasu/książki; 0 różnic pełnych ask ladders; największa bezwzględna różnica float64 0.
- 14 690 wartości cech S zależnych od book: 0 mismatch; max |Δ| 0.
- 1 260 wartości H z 60 pierwotnie przyjętych rynków: 0 mismatch i 0 niefinite; max |Δ| 0.
- Każdy z 4 odzyskanych rynków: 21/21 cech H porównano między niezależnym replayem i feature builderem poprawionego runnera; 84 wartości, 0 mismatch, max |Δ| 0. Stan i ścieżka pochodziły z niezależnego replayu.
- Event trace pokazuje czas source i receive, token/stronę, typ, cenę/rozmiar, accepted/ignored/conflict, stan przed/po, oraz wkład do depletion. Zapisano także synthetic endpoint counterexample.

## Gate feasibility przed dopasowaniem

| Warunek | Po audycie | Minimum | Wynik |
| --- | --- | --- | --- |
| Rynki oceny | 637 | 2 000 | FAIL |
| Obserwowane dni | 22 | 20 | PASS |
| Niepuste foldy | 2 | 2 | PASS |
| Udostępnione etykiety | 1 760 / 2 279; klasy 0 i 1 | ≥1 000 w każdym foldzie | PASS |
| Znane q=$5 po obu stronach | 908 / 1 207 | ≥1 000 w każdym foldzie | FAIL |

| Fold | Rynki oceny | Dni | Dostępne etykiety treningowe | Klasy | q=$5 po obu stronach | Labels | q=$5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fold 1 | 520 | 12 | 1 760 | 0, 1 | 908 | PASS | FAIL |
| Fold 2 | 117 | 10 | 2 279 | 0, 1 | 1 207 | PASS | PASS |

Gate ogólny: **FAIL**. Brakuje 1 363 rynków oceny względem progu 2 000 oraz 92 kwalifikowanych treningowych rynków q=$5 w foldzie 1. To luki liczebności, nie prognoza liczby możliwych do odzyskania rynków. Zbiory foldów mogą się nakładać. Nie uruchomiono fit ani ekonomiki.

## Wcześniejsze źródła i ograniczenia historii

Użyty lokalny oficjalny indeks obejmuje 21 962 market starts od 2026-04-15T17:05:00+00:00 do 2026-06-30T23:55:00+00:00. W lokalnych indeksach T−60/T−59 nie ma wierszy przed tym startem. RAW OOF ma 296 590 skończonych wierszy z 2023-06-20 do 2026-04-15 17:00Z; wszystkie są nominalnie T−60 względem startu, ale żaden nie ma zarejestrowanego `prediction_available_at_utc`; T−59 opiera się na założeniu 1 sekundy, a nie zmierzonym czasie. Przecięcie z wcześniejszym oficjalnym indeksem w dostarczonych plikach wynosi 0, lecz pełnego wcześniejszego indeksu globalnego nie zweryfikowano. Wartość 0 wpisana historycznie w runnerze nie dowodzi braku takich rynków.

Źródłowy replay ma 1 856 oczekiwanych godzin: 1 853 plików zweryfikowano SHA-256, w tym 1 409 niepustych i 444 pustych. Brakują 2026-06-11T04, T05 i T06; wszystkie trzy są poza oknami 26 h każdego z 1 469 badanych rynków. W ich oknach nie było brakujących partycji. Pełny indeks i archiwum sprzed 15 kwietnia oraz ich przecięcie S+H+RAW pozostają niezweryfikowane.

Backfill wymaga osobnego, źródłowo zweryfikowanego oficjalnego indeksu, historii wiadomości book/price-change od inicjalizacji potrzebnej dla H oraz kompatybilnych RAW OOF i T−60/T−59 drabin z potwierdzonym czasem dostępności. Należy hashować każdą godzinę, rozliczać puste partycje i braki, a następnie zmierzyć przecięcie S+H+RAW i poprawnych targetów. Dostępne pliki nie pozwalają oszacować liczby odzyskiwalnych rynków; masowego backfillu nie wykonywano.

## Koszt, walidacja i wykonane etapy

Niezależny replay: 7 949 784 wierszy zdarzeń z 1 242 757 525 bajtów zweryfikowanych partycji; 1 469 rynków, 4 workery, 767.9 s, 10,353 wierszy/s. Peak working set nie został wiarygodnie odczytany i pozostaje null; nie przedstawiam pojedynczego odczytu jako peak RAM. 1 853 partycji plików zweryfikowano hashem; brakujące trzy pozostawały poza próbką.

- Wykonano: inwentaryzację hashy i checkpointów; rekonstrukcję per-market/per-horizon dla 6 796 rynków; deterministyczny wybór 57 rozłącznych klas/okresów do 20 na klasę plus wszystkie 994 candidate IDs; niezależne odtworzenie dla 1 469 rynków; test syntetyczny; sprawdzenie recovered features; zamrożony gate przed fit.
- Nie wykonano: fit A/B/C/D; tuning; scoring/PnL/ROI/bootstrap; odczyt lub scoring wierszy rynków holdout; historyczny backfill; live settings; order submission.

Wyniki testów, kompilacji i diff-check z tej rewizji zapisano w historii commitów i w końcowym podsumowaniu audytu.

## Status zakresu

`models not fitted`; `economics not scored`; `holdout market rows not read/scored`; `live unchanged`; `orders not submitted`.

Pliki szczegółowe: sample plan, source partition inventory, worker traces/checkpoints, independent endpoint states, independent feature table, candidate recovery table, counterexample, comparison, gate i schematy/hashes są wskazane w wersjonowanym manifeście.
