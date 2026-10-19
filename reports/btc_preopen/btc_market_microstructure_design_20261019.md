# Projekt testu mikrostruktury Polymarket — 2026-10-19

## Decyzja i zakres

**Dane pozwalają na jeden nowy, ograniczony eksperyment warunkowy na stanie rynku:** kontrola z informacją rynku i sześcioma cechami mikrostruktury kontra ten sam model i te same wiersze z dodatkową informacją RAW BASE70. Projekt nie sprawdzi jednak wieku konkretnego najlepszego kwotowania ani asymetrii głębokości bid/ask. Dostępne cechy pozwalają zbadać spread, głębokość ask, koszt zakupu $5 oraz niespójność sumy midów.

Hipoteza o mikrostrukturze powstała po wcześniejszym oglądaniu holdoutu. Wynik ewentualnego testu będzie historyczną oceną development, nie niezależnym potwierdzeniem. Nie użyto etykiet ani PnL holdoutu do wyboru cech lub okresu. Nie trenowano modelu, nie liczono jakości ani ekonomii i nie zmieniano ustawień live.

## Audyt pochodzenia snapshotu i jego walidacji

Snapshot T−60 zapisuje `run_btc_preopen_t59_reassessment.py`, który używa stanu rekonstruowanego przez `run_btc_preopen_economic_replay.py::_snapshot`. Rekonstrukcja przetwarza zachowane zdarzenia książki w kolejności odbioru, śledzi osobno czas źródłowy i czas lokalnego odbioru, a przy braku bezpośredniej książki może utworzyć stronę komplementarną z drugiego tokenu. `archive_gap_after_up_book_init` i `archive_gap_after_down_book_init` wynikają z kontroli pokrycia godzin źródłowych po inicjalizacji książki. W badanym pliku wszystkie zapisane czasy wejścia T−60 zgadzają się z `market_start_utc - 60 s`.

Lokalna tabela T−60 ma 21 962 wiersze w development. Jej pola i ich rzeczywista dostępność:

| Informacja | Pola / semantyka | Dostępność w T−60 development | Wniosek |
|---|---|---:|---|
| Najlepszy bid i ask UP/DOWN | `up_best_bid/ask`, `down_best_bid/ask`; ceny mogą być komplementowane, co sygnalizują `*_quote_complemented` | 16 597 wierszy z cenami każdej strony; 16 596 oznaczone `has_full_snapshot` | Ceny do spreadu są dostępne, ale cechy proponowane są tylko dla dwóch natywnych, pełnych książek. |
| Ilość na najlepszym poziomie | `up_best_ask_size_shares`, `down_best_ask_size_shares` | 16 597 na każdej stronie ask | Ilości ask są zapisane. Ilości bid nie ma. |
| Zapisana głębokość | `up_ask_levels`, `down_ask_levels` zawierają listy `[price, shares]`; nie zapisano list bid | 16 597 niepustych list na każdej stronie; maksymalnie 74 poziomy UP i 71 DOWN; `depth_5usd_valid_both_sides=true` w 16 476 wierszach | Można wycenić zakup $5 z drabinki ask. Nie można policzyć pełnej głębokości bid ani asymetrii bid/ask. |
| Czas aktualizacji książki | `*_ask_age_seconds` i `*_ask_received_age_seconds` to wiek ostatniego zdarzenia strony ask według czasu źródłowego i odbioru | po 16 597 wierszy na stronę | To nie są timestampy zmiany aktualnego najlepszego poziomu. Brak wieku strony bid i timestampów aktualizacji konkretnych cen. Nie używam tych pól jako wieku kwotowania. |
| Luki źródłowe | `archive_gap_after_*_book_init`; plus pokrycie godziny wejścia i okna świeżości | flagi istnieją; 45 luk UP i 44 DOWN | Wymagam znanego pokrycia i braku luki po inicjalizacji. |
| Przyszłe zdarzenia | `no_future_event_at_entry`, `no_future_source_event_at_entry` | obie flagi istnieją | Wymagam obu flag prawdziwych. |
| Crossing / locked | `up/down_bid_ask_crossed`, `up/down_bid_ask_locked` | crossing UP 5 733, DOWN 5 336; locked UP 2 677, DOWN 2 788 | Crossing jest wykluczany przez istniejące `quote_valid`; osobne flagi zostają w tabeli diagnostycznej. |
| Niejednoznaczna kolejność | `up/down_ask_order_ambiguous` opisuje aktualny stan ask; `market_equal_source_timestamp_price_change_count`, `market_out_of_order_price_change_count`, `market_cross_side_late_price_change_count` są licznikami zdarzeń przed T−60 | flagi ask i liczniki istnieją; 9 528 snapshotów ma co najmniej jeden historyczny konflikt timestampu, 2 919 out-of-order i 2 007 cross-side-late; brak aktualnej, osobnej flagi niejednoznaczności bid | Kwalifikuję tylko bezpośrednie snapshoty z niejednoznacznością obu asków równą false. Liczników całej historii nie traktuję jako bieżącej niejednoznaczności. Brak pełnej kontroli bieżącego porządku bid. |
| Resety książki | `market_book_snapshot_count` liczy snapshoty zdarzeń; inicjalizacje mają receive timestamp | licznik jest, per-token flagi resetu brak | Nie da się odróżnić późniejszego resetu od inicjalizacji na podstawie zapisanej flagi. |

`quote_valid` w `_snapshot_features` sprawdza poprawność obu BBO, dostępność okna źródła, brak zdarzeń przyszłych i brak/znajomość luk. Do `p_market` potrzebne są obie strony. Dla każdej liczy się `mid=(bid+ask)/2`, a następnie `p_market=mid_UP/(mid_UP+mid_DOWN)`. Brak jednej strony daje brak `p_market`; nie wstawia się 0,5. Dzielenie przez sumę normalizuje midy i usuwa osobną informację, czy `mid_UP+mid_DOWN` odbiega od 1. Dlatego `mid_sum_minus_one` jest nową cechą.

W development `quote_valid` jest prawdziwe w 10 840 snapshotach; po wszystkich dodatkowych wymaganiach źródła, braku przyszłych zdarzeń i luk istniejące `p_market` przechodzi w 10 794 wierszach OOF.

## Tabela cech development

Tabela powstała wyłącznie z zapisanego RAW BASE70 OOF i T−60 snapshotów. Odczyt wybrał kolumny wejściowe bez `target_*`, rozliczeń ani pól T−59. Join wykonano po unikalnym `market_start_utc`; wszystkie 21 962 snapshoty są dokładnie w T−60. Dziewięć z 21 970 wierszy RAW OOF nie ma odpowiadającego snapshotu. Tabela nie zawiera targetu, etykiet oficjalnych ani wyniku transakcji.

Wartości cech mikrostruktury zapisano tylko dla natywnej, pełnej pary książek, prawidłowego `p_market`, znanego pokrycia bez luk i przyszłych zdarzeń, bez crossingu i bez aktywnej niejednoznaczności ask. Wiersz wspólny wymaga wszystkich sześciu cech oraz RAW i `p_market`. Braków nie imputuje się; w późniejszym teście oba warianty użyją tej samej maski.

| Cecha | Dokładny wzór | Pola źródłowe | Dostępność | Jednostka | Braki w 21 970 OOF | Obsługa braku |
|---|---|---|---|---|---:|---|
| `market_logit` | `logit(clip(p_market,1e-6,1-1e-6))` | Istniejące `p_market` z T−60 | T−60 | bezwymiarowa | 11 176 (50,9%) | Brak, gdy `p_market` jest nieważne; wspólna maska. |
| `spread_up` | `up_best_ask - up_best_bid` | T−60 ceny BBO UP | T−60 | punkty prawdopodobieństwa | 14 960 (68,1%) | Bez imputacji; usuń z obu wariantów przez wspólną maskę. |
| `spread_down` | `down_best_ask - down_best_bid` | T−60 ceny BBO DOWN | T−60 | punkty prawdopodobieństwa | 14 960 (68,1%) | Jak wyżej. |
| `up_down_ask_depth_imbalance` | `(up_best_ask_size_shares - down_best_ask_size_shares) / (up_best_ask_size_shares + down_best_ask_size_shares)` | Ilość na najlepszym ask UP/DOWN | T−60 | bezwymiarowa, zakres `[-1,1]` | 14 960 (68,1%) | Jak wyżej; to asymetria asków między tokenami, nie bid/ask. |
| `slippage_5_up` | `VWAP($5 po up_ask_levels) - up_best_ask` | Pełna zapisana drabinka ask UP | T−60 | punkty prawdopodobieństwa | 14 960 (68,1%) | Brak, gdy zapisana głębokość nie wystarcza na całe $5; bez proxy. |
| `slippage_5_down` | `VWAP($5 po down_ask_levels) - down_best_ask` | Pełna zapisana drabinka ask DOWN | T−60 | punkty prawdopodobieństwa | 14 960 (68,1%) | Jak wyżej. |
| `mid_sum_minus_one` | `((up_best_bid+up_best_ask)/2 + (down_best_bid+down_best_ask)/2) - 1` | T−60 ceny BBO UP/DOWN | T−60 | odchylenie sumy od 1, w punktach prawdopodobieństwa | 14 960 (68,1%) | Bez imputacji; wspólna maska. |
| `raw_minus_market_logit` | `logit(clip(p_RAW,1e-6,1-1e-6)) - logit(clip(p_market,1e-6,1-1e-6))` | RAW BASE70 OOF i `p_market` z T−60 | OOF dla nominalnego T−60; bez snapshotu T−59 | bezwymiarowa | 11 176 (50,9%) | Brak, gdy któregokolwiek prawdopodobieństwa brak; wspólna maska. |

RAW BASE70 OOF ma 21 970 wierszy w przedziale `2026-04-15T17:05Z`–`2026-06-30T23:50Z`. Zapisane `p_market` jest dostępne w 10 794 wierszach (49,1%); wszystkie RAW OOF mają prawdopodobieństwo. Wszystkie sześć cech mikrostruktury jest wspólnie dostępnych w 7 010 wierszach (31,9% RAW OOF; 65,0% wierszy z `p_market`). Rozkład pokrycia jest nierówny: kwalifikujące się wiersze to 911 w kwietniu, 3 160 w maju i 2 939 w czerwcu; tylko 58 przypada po `2026-06-12`. Żaden okres nie został odcięty na podstawie jakości prognoz, etykiet ani PnL.

**Porównanie ze starym M0/M1:** `logit(p_market)` oraz `raw_minus_market_logit` odpowiadają informacjom już testowanym przez M0/M1. Wcześniej nie używano spreadów, ilości ask, kosztu $5 z pełnej drabinki ani `mid_sum_minus_one` jako predyktorów. Nowość to stan książki warunkujący porównanie, a nie nowa definicja różnicy RAW–rynek. Wiek konkretnego kwotowania, asymetria bid/ask i wpływ resetów pozostają nieweryfikowalne z zapisanej tabeli.

## Jeden z góry ograniczony eksperyment

Konfiguracja jest zapisana w `configs/research/btc_market_microstructure_design_20261019.json`. Zastosuje się tylko tę jedną procedurę:

1. **Cel:** oficjalne `target_polymarket_up`. Etykieta dostępna o `max(resolved_at_utc, market_start_utc + 5 min) + 60 s`.
2. **Chronologia:** jeden fit na prefixie `2026-04-15T17:05Z`–`2026-05-15T00:00Z`; test na wszystkich kolejnych wierszach do wyłącznego cutoffu `2026-07-01T00:00Z`. Pierwsza decyzja testowa to `2026-05-14T23:59Z`; do fitu wchodzą wyłącznie oficjalne etykiety zwolnione ściśle przed tą chwilą. Wspólna maska testowa dodatkowo wymaga oficjalnej etykiety dostępnej do cutoffu. Zmierzone liczby 1 927 w prefixie i 5 083 w teście są kandydatami przed odjęciem brakujących/niedostępnych etykiet.
3. **Kontrola:** `StandardScaler` dopasowany na prefixie i regresja logistyczna L2, `C=1`, `solver=lbfgs`, `max_iter=1000`, bez wag klas. Wejścia: `market_logit` oraz sześć powyższych cech mikrostruktury.
4. **Challenger:** ten sam pipeline, ustawienia, prefix i dokładnie te same wiersze, plus tylko `raw_minus_market_logit`. Każdy model dopasowuje się jeden raz; w okresie testowym nie ma refitów ani doboru parametrów.
5. **Jakość:** główny kontrast to challenger minus control dla log loss; dodatkowo Brier i AUC. Przedziały sparowane na wspólnych 3-dniowych blokach UTC, 2 000 replikacji, seed `20261019`. Wyniki rozpisane łącznie i miesięcznie z liczebnością; brak pokrycia pod koniec czerwca pozostaje widoczny.
6. **Ekonomia:** niezmieniona historyczna polityka z konfiguracji BASE70 i audytu przepływów `a5fd05e`: osobny zakup notionalu $5, próg EV netto 1% żądanego notionalu, sygnał z informacji T−60 i wykonanie na zapisanej książce T−59, historyczny tryb opłat, głębokość i cutoff. Control i challenger niezależnie wybierają stronę i transakcje na wspólnym kalendarzu; T−59 służy tylko do wyceny/wykonania już wybranej decyzji. Raport obejmie fills/settlements, notional, debet, opłaty cash i ekwiwalent tokenowy, PnL, PnL/notional, PnL/debet, EV na tych samych settled fillach, nierozliczone debety, no-trade PnL `$0`, wyniki miesięczne, koncentrację zysku i sparowane dzienne różnice.

Nie ma dodatkowych miksów, progów, seedów modelu ani wyboru okresu po wyniku. Model BASE70 pozostaje zapisanym OOF; nie będzie ponownego treningu. Rynek i mikrostruktura są silną kontrolą, więc kontrast odpowiada na pytanie, czy RAW wnosi coś ponad stan książki, a nie czy RAW pokonuje sam rynek.

## Wykonalność i koszt

Pomiar na lokalnej maszynie, pojedynczy przebieg, bez treningu:

| Etap | Dane | Czas |
|---|---:|---:|
| Odczyt kolumn T−60 dla development | 25 908 421 B | 0,276 s |
| Odczyt kolumn RAW OOF dla development | 11 166 225 B | 0,010 s |
| Join i budowa cech dla 21 970 OOF / 21 961 dopasowanych snapshotów | tabela 1 646 514 B | 1,266 s |
| Zapis Parquet Zstandard | 1 646 514 B | 0,037 s |

Szczytowy working set procesu wyniósł 801 394 688 B (około 0,75 GiB). Tabela ma 35 kolumn, nie zawiera labeli, i jest zapisana lokalnie w `data/analysis/polymarket/BTC/btc_market_microstructure_design_20261019/development_features.parquet`. Hashe wejść, kodu, konfiguracji i tabeli są w manifeście. Czasy pochodzą z jednego przebiegu z lokalnym cache systemu plików; nie są pomiarem zimnego odczytu.

Przyszłe dopasowanie to dokładnie dwa modele, każdy na maksymalnie 1 927 wierszach × 7 lub 8 wejść, i punktacja do 5 083 kandydatów. Przy założeniu jednego wątku CPU obliczenie modelu powinno zamknąć się poniżej minuty; to estymacja, nie pomiar. Tabela cech i odczyty są małe. Nie ma potrzeby trenować BASE70, korzystać z GPU, pobierać danych ani odtwarzać archiwum.

## Wnioski z wykonalności

- Zapisane dane **pozwalają** przetestować, czy RAW wnosi informację poza prawdopodobieństwo rynku i dostępny stan spreadu/ask, w tym potencjalny koszt natychmiastowego zakupu $5 oraz rozjazd sumy midów.
- Nie pozwalają sprawdzić pełnej asymetrii głębokości bid/ask ani wieku najlepszego bid/ask. Nie wolno zastąpić ich wiekiem ostatniego zdarzenia strony książki.
- Test będzie miał 5 083 kandydatów z pełnym zestawem wejść w okresie oceny, ale ich 3-dniowa blokowa struktura jest nierówna; tylko 58 przypada po 12 czerwca. W raporcie trzeba pokazać tę lukę i nie przedstawiać wyniku jako dowodu stabilności do końca development.
- RAW BASE70 zachował historyczny sygnał predykcyjny, lecz holdout nie wykazał dodatniej ekonomii. M1 poprawiał jakość prawdopodobieństw ogółem, ale z ujemnym PnL i gorszym PnL/notional nie kwalifikuje się do przyjęcia przy obecnej polityce. Nie planuje się kolejnych wariantów mieszania prognoz, progów ani seedów.

## Artefakty

- Konfiguracja i protokół: `configs/research/btc_market_microstructure_design_20261019.json`.
- Budowa tabeli: `build_btc_market_microstructure_design_20261019.py` (bez argumentów CLI, bez fitu).
- Manifest i pomiar: `reports/btc_preopen/btc_market_microstructure_design_20261019_manifest.json`.
- Tabela: `data/analysis/polymarket/BTC/btc_market_microstructure_design_20261019/development_features.parquet`.
- Testy wzorów i przyczynowości: `tests/test_btc_market_microstructure_design_20261019.py`.
