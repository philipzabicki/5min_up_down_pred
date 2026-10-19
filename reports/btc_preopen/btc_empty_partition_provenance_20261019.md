# Pochodzenie pustych partycji Polymarket BTC 5m — 13–28 czerwca 2026

**Wniosek:** **przyczyna nieustalona z lokalnych artefaktów**. Potwierdzone zero wierszy dotyczy lokalnego wyniku po selekcji rynków i czasu, nie oryginalnego pliku archiwalnego. Oryginalne bajty godzinowych plików PMXT z badanego okresu nie są przechowywane lokalnie.

Audyt wykorzystał tylko zapisane pliki, manifesty, checkpointy i kod. Nie pobierano danych, nie uruchamiano treningu, eksperymentu C/X ani replayu.

## Tabela dowodów

| Ścieżka | Etap przetwarzania | Liczba wierszy | Źródło | Status weryfikacji |
|---|---|---:|---|---|
| `https://r2v2.pmxt.dev/polymarket_orderbook_2026-06-{13..28}T{00..23}.parquet` | Oryginalne archiwum godzinowe | Nieznana lokalnie; checkpoint podaje 16–69 grup Parquet na plik, nie liczbę wierszy | URL-e i wyniki odczytu w `data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261007/pmxt/hour_checkpoint.json` | Bajtów źródłowych brak. Wszystkie 384 wpisy miały co najmniej jedną odczytaną grupę i status `no_target_events_before_entry`. |
| `data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/pmxt_extended/archive_checksums/pmxt_v2_SHA256SUMS.txt` | Indeks sum źródeł, nie kopia źródła | 384 reklamowane sumy dla tych godzin | Lokalny indeks PMXT v2 z października | Sumy są metadanymi wydawcy; stare checkpointy nie wiążą ich z bajtami URL-i `r2v2.pmxt.dev`. Nie potwierdzają zawartości użytej podczas ekstrakcji. |
| `data/raw/polymarket/obadiaha/11793901f0ac89c5a6c51123a6ccd29a3aaf8f4c/orderbooks/` oraz `data/raw/polymarket/kachoio/42d917dc8e3205dde8ac909792af0cce2d715c9f/` | Lokalna kopia innych źródeł | Brak lokalnych plików dla 13–28 czerwca; liczba źródłowych wierszy: nie dotyczy | Manifesty lokalnych zbiorów Hugging Face | `obadiaha` ma 21 dziennych plików orderbooków do 19 marca; `kachoio` ma ticki do 18 maja. Żaden nie jest kopią badanych godzin PMXT. Brak pliku nie jest liczbą wierszy równą zero. |
| `data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261007/pmxt/event_parts/2026-06-13T*.parquet` | Wynik ekstrakcji po selekcji | 384 pliki, 0 wierszy łącznie | `hour_checkpoint.json`: `selected_event_rows=0`, status `no_target_events_before_entry` | Lokalny Parquet i SHA-256 zweryfikowane. To pierwsza warstwa z zapisaną liczbą wierszy; nie dowodzi pustki oryginałów. |
| `data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/pmxt_extended/event_parts/2026-06-13T*.parquet` | Partycje wejściowe replayu | 384 pliki, 0 wierszy łącznie | `t59_replay_dependencies.json`, `archive_manifest.json` oraz metadane Parquet | Każdy plik ma 2 577 bajtów. SHA-256 wszystkich 384 plików jest zgodny z odpowiadającym wynikiem ekstrakcji. Replay otrzymał już puste wyniki selekcji. |
| `data/analysis/polymarket/BTC/preopen_research_20261011/book_replay_t60_t59/t60_decision_books_with_source_coverage.parquet` | Snapshot T−60 | 4 602 snapshoty dla 4 608 startów z OOF; `market_event_count=0` w każdym | Zapisany snapshot T−60 oraz lokalny `B0_decision_oof.parquet` | Wszystkie 4 602 czasy wejścia są równe `market_start_utc − 60 s`. Sześć pozostałych startów nie ma rekordu Gamma w indeksie rynku. Snapshot potwierdza brak zdarzeń w wejściu replayu, nie stan oryginalnego archiwum. |

Pełna tabela godzinowa z URL-em, grupami Parquet, ścieżkami, rozmiarami i hashami obu lokalnych partycji jest w [CSV — 384 godziny](btc_empty_partition_provenance_20261019_hourly.csv). Puste pola dla liczby wierszy źródłowych oznaczają „niezapisane / brak lokalnych bajtów”, a nie zero.

`empty_source_partition_audit.json` zapisuje `archive_rows_read_or_replayed=0`, ale opisuje własną kontrolę jako „Parquet metadata only; no event rows were read or replayed”. Ta wartość nie jest liczbą wierszy oryginalnych plików.

## Gdzie znika liczba wierszy

Stary checkpoint zawiera 384 godziny, wszystkie ze statusem `no_target_events_before_entry`, jedną próbą i bez błędów retry. `row_groups_total` i `row_groups_read` są dodatnie dla każdej godziny; zakres wynosi 16–69. W 309 godzinach odczytano wszystkie grupy, w pozostałych odczytano część kandydatów. Nie zapisano liczby surowych wierszy, liczby dokładnych dopasowań `market` ani liczby wierszy odrzuconych przez cutoff czasu.

Dlatego lokalne dowody umieszczają pierwsze **potwierdzone zero** między odczytem grup Parquet ze źródła a zapisem filtrowanej partycji. Nie rozstrzygają, czy w oryginale nie było zdarzeń dla tych warunków, czy dokładne dopasowanie warunku rynku nie znalazło wierszy, czy wszystkie dopasowane zdarzenia odpadły na granicy czasu. **Przyczyna nieustalona z lokalnych artefaktów.** Nie ma podstaw, aby nazwać to potwierdzonym brakiem danych źródłowych ani aby wykluczyć błąd selekcji.

384 oznacza partycje godzinowe czasu źródła, a nie 384 rynki. W kalendarzu OOF są 4 608 starty pięciominutowe w tych 16 dniach:

| Dzień UTC | Starty OOF | Snapshot T−60 | Zdarzenia w snapshotach | Bez rekordu rynku Gamma |
|---|---:|---:|---:|---:|
| 2026-06-13 | 288 | 288 | 0 | 0 |
| 2026-06-14 | 288 | 288 | 0 | 0 |
| 2026-06-15 | 288 | 288 | 0 | 0 |
| 2026-06-16 | 288 | 288 | 0 | 0 |
| 2026-06-17 | 288 | 283 | 0 | 5 |
| 2026-06-18 | 288 | 288 | 0 | 0 |
| 2026-06-19 | 288 | 288 | 0 | 0 |
| 2026-06-20 | 288 | 287 | 0 | 1 |
| 2026-06-21 | 288 | 288 | 0 | 0 |
| 2026-06-22 | 288 | 288 | 0 | 0 |
| 2026-06-23 | 288 | 288 | 0 | 0 |
| 2026-06-24 | 288 | 288 | 0 | 0 |
| 2026-06-25 | 288 | 288 | 0 | 0 |
| 2026-06-26 | 288 | 288 | 0 | 0 |
| 2026-06-27 | 288 | 288 | 0 | 0 |
| 2026-06-28 | 288 | 288 | 0 | 0 |

Sześć startów bez rekordu Gamma to `2026-06-17T20:05:00Z`, `20:10:00Z`, `20:15:00Z`, `20:20:00Z`, `20:25:00Z` oraz `2026-06-20T17:40:00Z`. Pozostałe 4 602 mają rekordy rynku i etykiety oficjalne, ale nie mają obserwowanych zdarzeń w zapisanych partycjach.

## Mapowanie, schemat, czas i obsługa błędów

- Ekstraktor selekcjonuje kolumnę `market` (identyfikator warunku/rynku) i cutoff `timestamp_received`; filtr nie używa `asset_id` ani tokenów UP/DOWN. Mapowanie tokenów jest stosowane przy odtwarzaniu książek. Lokalny indeks ma 4 602 wiersze dla tych rynków, niepuste i różne identyfikatory UP/DOWN. Ponieważ wejściowe partycje są puste, zgodności czerwcowych `asset_id` z tokenami nie da się sprawdzić.
- Schemat zachowanych pustych Parquetów ma `timestamp_received` i `timestamp` jako `timestamp[ms, tz=UTC]`, `market` jako `fixed_size_binary[66]` i `asset_id` jako string. Kod ekstrakcji wymaga kolumn zdarzeń; brak wymaganej kolumny prowadzi do retry/błędu technicznego, nie do `no_target_events_before_entry`. Zapisany czerwcowy checkpoint nie zawiera błędów schematu. Nie zawiera jednak oryginalnego schematu/wierszy poza typami skopiowanymi do pustego wyniku.
- Cutoff opiera się na czasie odbioru w milisekundach UTC. W 4 602 snapshotach T−60 czas wejścia zgadza się z rynkiem co do sekundy. Jako kontrolę sąsiedniej partycji sprawdzono istniejący plik `2026-06-30T00.parquet`: 154 750 wierszy, 297 warunków, wszystkie `asset_id` mapują się do UP/DOWN danego rynku, a czasy odbioru są przed właściwym cutoffem. To nie zastępuje kontroli oryginalnych plików z luki.
- Dla 384 godzin checkpoint ma zero retry błędów, jedną próbę i status `no_target_events_before_entry`, odrębny od `archive_file_not_found` i `technical_error`. Nie wskazuje to na zgłoszony błąd transportu.
- Puste wyniki są zapisywane jako poprawne partycje i mogą być ponownie użyte, jeśli lokalny plik istnieje i jego hash zgadza się z checkpointem. To sprawdza wynik, nie niezmienność zdalnego źródła. Historyczny checkpoint nie zapisuje hashy/ETag oryginalnych obiektów ani hashy kodu ekstrakcji. Lokalny, późniejszy indeks ma reklamowane SHA-256 dla wszystkich 384 nazw plików, lecz oryginalnych bajtów brak, a stare wyniki nie są z tymi hashami powiązane. Możliwe jest więc ponowne użycie starego pustego wyniku po zmianie obiektu pod tym samym URL-em; nie ma dowodu, że taka zmiana nastąpiła.

W danych BASE70 jest też niespójność uzasadnienia pominięcia: dla 4 602 rynków pola `archive_entry_hour_available=True`, `archive_history_complete=True`, `t59_snapshot_available=True`, `snapshot_market_event_count=0` i `ask_data_status=no_archive_events_observed_by_t59` współistnieją z `fixed5_skip_reason=archive_hour_missing_at_entry`. Ledger decyzji RAW zapisuje te wiersze jako `no_priceable_native_ask_side`. Pole powstaje w `run_btc_preopen_t59_reassessment.py:2708`; zapisane `archive_hour_missing_at_entry` nie wynika z zapisanych flag i należy skorygować/regenerować tę diagnostykę. Obecny kod nie ma osobnej etykiety dla istniejącego snapshotu bez zdarzeń. Sam ledger nie pokazuje zleceń ani PnL dla tych wierszy.

## Wpływ na BASE70 oraz M0/M1

- **Jakość prognoz BASE70:** jej zapisany zbiór jakości nie wymaga `p_market`. Obejmuje 4 602 rynki z luki z oficjalnymi etykietami (`2 322` UP, `2 280` DOWN); sześć startów bez rynku Gamma nie ma etykiety. Ta luka nie usuwa tych 4 602 wierszy z niezależnej oceny RAW BASE70.
- **Ekonomia BASE70:** zapisane decyzje obejmują cały kalendarz, ale 4 602 rynki są pominięte z powodu braku priceable asków, a sześć z powodu braku rekordu Gamma. Raport BASE70 rozdziela globalnie brak priceable danych od progu EV i braku rynku, lecz nie oznacza osobno tych 16 dni w dziennym PnL. Wiersze dzienne z zerowym PnL/obrotem należy tu czytać jako brak dostępnego wejścia, nie jako dowód, że polityka oceniła rynek i zdecydowała „nie handlować”.
- **M0/M1 development:** zapisane `development_predictions.parquet` ma 4 608 wierszy z tych dat, ale `p_market`, `p_m0` i `p_m1` są puste. 4 602 oficjalnie etykietowane rynki nie weszły do wspólnej próby jakości 9 694 wierszy ani kalendarza ekonomicznego 9 698 decyzji; sześć pozostałych nie ma rynku/etykiety Gamma. Zamrożone wiersze dopasowania M0/M1 (10 790) nie zawierają dat z luki. Dzienny plik PnL pokazuje 16 dni jako zera, mimo że ledger M0/M1 nie ma dla nich decyzji; raport nie rozróżnia tych zer od dnia z dostępnymi danymi i brakiem transakcji.
- **Holdout M0/M1 od lipca:** zakres samej oceny zaczyna się 1 lipca, więc czerwcowe snapshoty nie są brakującymi wejściami holdoutu. Współczynniki dopasowano na dostępnej próbie rozwojowej; ta próba nie obejmuje 4 602 rynków bez `p_market`.

Nie znalazłem konkretnego błędu księgowania PnL. Zachowuję istniejące PnL jako wyniki dostępnych wierszy/wykonań; nie przeliczam go i nie dopisuję kontrfaktycznych transakcji. BASE70 raportował dostępność i powody pominięcia globalnie, ale nie tę lukę per dzień. Raport M0/M1 pokazywał wspólną próbę, lecz dzienne zera nie wyjaśniały, że w tych dniach nie było wejściowych decyzji.

## Wymagana naprawa

1. W manifestach ekstrakcji zapisywać osobno: liczbę wierszy źródłowych, dokładne dopasowania `market`, dopasowania po cutoffie oraz przyczynę pustego wyniku. Do cache pustych godzin dołączyć hash źródłowego obiektu z indeksu (lub stabilny ETag/Last-Modified). Bez wersji źródła oznaczać pusty wynik jako nieweryfikowalny i nie uznawać samego SHA pustego Parquetu za potwierdzenie źródła.
2. W `run_btc_preopen_t59_reassessment.py:2708` dodać przed gałęzią `t59_snapshot_available & fixed5_priceable_side_count == 0` warunek dla istniejącego T−59 snapshotu z `snapshot_market_event_count == 0`; emitować `no_archive_events_observed_by_t59`, a `archive_hour_missing_at_entry` rezerwować dla nieobecnego/nieprzetworzonego źródła. W zapisanym kalendarzu ponownie wyliczyć tylko tę diagnostyczną kolumnę z istniejących flag. Dodać liczbę decyzji/dostępnych snapshotów do dziennego PnL M0/M1, aby zero wejściowych wierszy nie wyglądało jak dzień bez transakcji.
3. Aby rozstrzygnąć źródłową przyczynę, odzyskać i zweryfikować oryginalne 384 pliki, po czym ponowić wyłącznie ich selekcję do audytu. Nie trzeba do tego powtarzać pełnego replayu. Tego kroku nie wykonano w tym zadaniu zgodnie z zakresem.

## Artefakty sprawdzone

- `data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261007/pmxt/hour_checkpoint.json`
- `data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/pmxt_extended/archive_manifest.json`
- `data/analysis/polymarket/BTC/preopen_research_20261011/book_replay_t60_t59/t59_replay_dependencies.json`
- `data/analysis/polymarket/BTC/btc_market_microstructure_development_20261019/availability/empty_source_partition_audit.json`
- `data/analysis/polymarket/BTC/btc_base70_information_value_20261018/common_execution_calendar.parquet`
- `data/analysis/polymarket/BTC/btc_market_base70_incremental_20261019/development_predictions.parquet`
- `run_btc_preopen_pmxt_extract.py`, `run_btc_preopen_t59_reassessment.py`, `run_btc_preopen_economic_replay.py`
