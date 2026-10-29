# E0042 / T0014 — D holdout local source inventory

Data inwentaryzacji: 2026-10-29 (Europe/Warsaw). Zakres jest metadata-only zgodnie z instrukcją do tego zadania; źródła/indeksy przejrzano wyłącznie w jawnie wskazanych lokalizacjach poniżej.

## Odpowiedź

**Czy lokalne źródła pozwalają przystąpić do przygotowania oceny?** Można zaplanować i później przygotować częściowe, źródłowe wejście. Nie można obecnie zbudować kompletnego, pełnego pokrycia par T−60/T−59 ani wykazać pełnej ekonomiki dla całego holdoutu: lokalny indeks potwierdza 28 222 rynki z 28 224 oczekiwanych slotów, a archiwum zdarzeń księgi ma 68-godzinną lukę i 1 115 rynków z niekompletnym oknem historii.

**Co mamy?** Gotowy i hash-zgodny bundle D E0041; Gamma i PMXT techniczne indeksy; 64/70 pól bazowych w szerokim źródle cech oraz pozostałe sześć taker_Z/V w lokalnym produkcie przepływów; pliki OHLCV i historyczny warmup; 4 137 godzinowych partycji zdarzeń PMXT poza zidentyfikowanymi lukami. Metadane przepływów obejmują wymagany okres, ale treść wszystkich danych pozostaje `NOT_VERIFIED` w tym audycie.

**Czego brakuje?** Nie ma przygotowanych `evaluation_rows.parquet` i `evaluation_input_manifest.json`; dynamiczne stany memory/ESN trzeba odtworzyć przyczynowo z developmentowego prefiksu; brakujące są dokładnie godziny `[2026-08-15T10:00:00Z, 2026-08-18T06:00:00Z)`; dwa sloty nie mają pary w lokalnych indeksach; treści BBO, świeżości, opłat i jakości strumienia nie sprawdzono.

**Jeden konkretny następny krok:** osobne, wąskie uzupełnienie/proweniencja źródła pełnej księgi dla 68 godzin `[2026-08-15T10:00:00Z, 2026-08-18T06:00:00Z)`, a następnie ponowny metadata-only audyt luk. Nie pobierano niczego w tym zadaniu. Dwa unmatched sloty pozostają osobną nierozstrzygniętą kwestią.

E0041 manifest: [final_D_bundle_manifest.json](../../data/analysis/polymarket/BTC/btc_D_holdout_preparation_20261028/final_D_bundle_manifest.json), SHA-256 `7715419ce5efbce8405db861fb3a67c481038c07f11eec019137f14528f0f97d`. Stan z manifestu: `D_BUNDLE_READY`, `activated=false`, `evaluation_authorized=false`, `HOLDOUT_NOT_RUN`, odczytanych holdout rows 0, wygenerowanych predykcji 0. Zachowano run `189049366841f6d025e3d9b080bb69dd0e441bb5638e451708f27c0635754c44`, dziewięciu członków po 86 cech, 78 622 development OOF i 21 951 oficjalnych wierszy kalibracji. Bundle i modele nie były odczytywane ponownie poza manifestem.

## Granice audytu

Dozwolone projekcje dotyczyły wyłącznie struktur plików, rozmiarów, row/row-group counts, nazw schematu, statystyk czasu z footerów, zapisanych hashy oraz technicznych pól indeksów: `condition_id`, `market_start_utc`, token IDs i metadane obecności/źródła/kompletności archiwum. Nie czytano wartości outcome, payoutów, predykcji, cech, ofert, cen ani głębokości. Nie odczytano payloadu surowych eventów, świec, CSV ani dziennych JSON. Nie pobierano źródeł, nie tworzono wejść, predykcji, ksiąg, scoringu ani ekonomiki.

Wszystkie treści plików źródłowych mają status `NOT_VERIFIED` w tej inwentaryzacji. Historyczne manifesty i wcześniejsze raporty są oznaczone jako odziedziczone dowody, a nie świeża walidacja payloadu. Brak lokalnej pozycji w wymienionym indeksie oznacza tylko unmatched względem tego indeksu, nie dowód nieistnienia rynku globalnie.

## Kalendarz Gamma i pokrycie

Holdout to `[2026-07-01T00:00:00Z, 2026-10-07T00:00:00Z)`: 98 dni × 288 = **28 224** oczekiwane sloty pięciominutowe. Techniczna projekcja `gamma_markets.parquet` daje **28 222** unikalne starty; duplicate starts: 0. Oba rynki w technicznej projekcji mają token IDs. Projekcja PMXT `market_index.parquet` zawiera dokładnie te same 28 222 pary `(condition_id, market_start_utc)`. Oficjalne pola Gamma `target_polymarket_up` i `resolved_at_utc` są obecne w schemacie, ale ich wartości nie zostały odczytane.

Dwa sloty bez dopasowania w obu indeksach:

- `2026-07-12T02:25:00Z`
- `2026-09-03T15:20:00Z`

Nie wiadomo z tego audytu, czy slot nie miał rynku, czy lokalny indeks jest niepełny. Nie utworzono sztucznych rynków ani transakcji.

| Okres/reżim zapisany w protokole | Sloty oczekiwane | Gamma potwierdzone | Różnica względem kalendarza | Z historią archiwum PMXT kompletną |
|---|---:|---:|---:|---:|
| Single price `[2026-07-01, 2026-08-07)` | 10 656 | 10 655 | 1 | 10 655 |
| TWAP30 `[2026-08-07, 2026-08-14)` | 2 016 | 2 016 | 0 | 2 016 |
| TWAP60 `[2026-08-14, 2026-10-07)` | 15 552 | 15 551 | 1 | 14 436 |
| **Razem** | **28 224** | **28 222** | **2** | **27 107** |

| Miesiąc | Sloty oczekiwane | Gamma potwierdzone | PMXT z kompletnym oknem historii |
|---|---:|---:|---:|
| Lipiec | 8 928 | 8 927 | 8 927 |
| Sierpień | 8 928 | 8 928 | 7 813 |
| Wrzesień | 8 640 | 8 639 | 8 639 |
| Październik 1–6 | 1 728 | 1 728 | 1 728 |

Kalendarz single/TWAP jest zamrożoną granicą reżimów, nie oficjalną etykietą. Nie przeliczano rozbieżności Gamma–Binance.

## Mapa fizycznych źródeł

| Źródło (ścieżka względem repo) | Fizyczna obecność i metadane | Status / ograniczenie |
|---|---|---|
| `data/analysis/polymarket/BTC/btc_D_holdout_preparation_20261028/final_D_bundle_manifest.json` | 17 388 B; hash podany wyżej; manifest wskazuje dziewięć modeli D | Gotowy bundle; nie autoryzuje ewaluacji |
| `data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261007/gamma_markets.parquet` | 10 729 904 B, 80 127 wierszy, 1 row group; footer czasu 2025-12-18 04:25–2026-10-08 13:10 | 28 222 potwierdzone starty w holdoucie; etykiety i minimum share nieodczytane |
| `data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/pmxt_extended/market_index.parquet` | 7 486 947 B, 50 184 wiersze, 1 row group; footer 2026-04-15 17:05–2026-10-06 23:55 | Para technicznych ID zgodna z Gamma; 1 115 niekompletnych okien historii |
| `data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/pmxt_extended/archive_manifest.json` | Fizyczny JSON; cutoff 2026-10-07T00:00Z; PMXT v2/v3/AG6 i zakres full-venue-gap | Zapisuje lukę 68 h; manifest sam nie potwierdza poprawności treści |
| `data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261008/pmxt_extended/event_parts/` | 4 137 plików godzinowych, razem 5 296 270 309 B; nazwy od `2026-04-14T16.parquet` do `2026-10-06T23.parquet` | Payloadu eventów nie otwarto; fizyczny skan nazw potwierdza lukę i nie dowodzi BBO/fillu |
| `.../BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.parquet` | 4 757 016 837 B, 3 327 267 wierszy, 4 groups, 656 kolumn, Opened do 2026-10-06 23:59 | Schemat ma 64/70 BASE; sześć `taker_Z/V` dostarcza osobny produkt przepływów |
| `.../feature_dataset_with_causal_candidates.parquet` | 5 382 961 980 B, 3 186 145 wierszy, 25 groups, 654 kolumn, Opened do 2026-06-30 23:57 | 70/70 BASE w schemacie; developmentowy input do stanów, nie holdout |
| `.../D_FINAL_SELECTION_20251001/dynamic_features.parquet` | 443 011 214 B, 3 186 145 wierszy, 4 groups, 33 kolumny | Brak runtime state w Parquet |
| `.../D_FINAL_SELECTION_20251001/dynamic_features.checkpoint.json` | 3 089 B; identity, hash, liczba wierszy/kolumn, reguła resetu | Nie zawiera stanów reservoir/memory |
| `data/analysis/polymarket/BTC/btc_base70_raw_holdout_20261018/holdout_flow_features.parquet` | 30 736 264 B, 184 320 wierszy, 19 kolumn; Opened 2026-06-01–2026-10-06 23:59 | W schemacie wszystkie 6 wymaganych `taker_Z/V`; historyczny manifest gridu 184 320/184 320, ale treść NOT_VERIFIED |
| `.../source_completion/taker_flow_features_20260701_20261006.parquet` | 8 805 782 B, 141 113 wierszy, 7 kolumn; 2026-07-01–2026-10-06 23:52 | Węższy duplikat potrzebnego zestawu 6 pól; nie zastępuje pełniejszego źródła |
| `.../source_completion/raw_by_day/` | 98 dziennych JSON, 18 674 571 B według manifestu, 2026-07-01–2026-10-06 23:52 | Payloady JSON nieotwarte |
| `data/datasets/_tmp/binance_vision/um/klines/BTCUSDT1m/months/2026-06..09/` | Fizyczne CSV: 4 894 009; 5 016 863; 5 004 926; 4 862 487 B | Pliki obecne; zakres i jakość treści opierają się na manifestach historycznych, nie świeżym odczycie CSV |
| `data/datasets/_tmp/binance_vision/um/klines/BTCUSDT1m/days/2026-10-01/` | CSV obecny, 162 829 B | Payload nieotwarty |
| `.../btc_base70_raw_holdout_20261018/binance_daily_download_staging/` | Pięć ZIP-ów dla 2–6 października i pięć checksum sidecarów, razem 297 007 B; pojedynczy CSV member na ZIP | Lista ZIP members sprawdzona bez ekstrakcji; CSV payloady nieotwarte |
| T0009 `data/analysis/polymarket/BTC/btc_t0009_resolution_coverage_20261026_38a6221f0d90/pmxt_replay/t60_decision_books.parquet` i `t59_ask_ladders.parquet` | Po 51 458 wierszy; odpowiednio 11 867 882 B i 11 942 911 B; zakres do 2026-06-30 23:55 | Development-only, brak holdoutu |
| T0009 `.../btc_t0009_resolution_coverage_20261026_38a6221f0d90/coverage_manifest.parquet` | 11,273,579 B; prior manifest covers 105,118 development rows, 2025-07-01 through 2026-06-30 | Development-only; not holdout coverage |
| T0009 `.../btc_t0009_source_quality_continuation_20261026/source_quality_manifest.parquet` | 1,163,904 B; prior audit has 5,684 development rows, 2026-04-15 through 2026-06-30 | Development-only; content NOT_VERIFIED in this inventory |
| T0009 `.../pmxt_replay/t59_replay_manifest.json` | 2 818 B; prior replay manifest | Development-only |
| `data/raw/polymarket/obadiaha/11793901f0ac89c5a6c51123a6ccd29a3aaf8f4c/` | 25 plików, 135 000 143 B; wcześniejsze źródłowe zestawienie: 27 332 rynki | Indeksowane orderbooki tylko Jan/Mar 2026; alternatywa nie pokrywa holdoutu |
| `data/raw/polymarket/kachoio/42d917dc8e3205dde8ac909792af0cce2d715c9f/` | 4 pliki, 186 405 662 B; wcześniejsze zestawienie: 15 682 rynki / 4 704 518 ticków | Mar–Maj 2026; development-only |
| `.../pmxt_v2_archive_full_scenarios/market_index.parquet` | 9 407 wierszy, Apr 15–May 18 2026 | Development-only |
| `data/analysis/polymarket/BTC/btc_D_holdout_evaluation_20261028/evaluation_rows.parquet` i `evaluation_input_manifest.json` | Oba pliki/ścieżka nie istnieją lokalnie | Brak produktu przygotowawczego; nie oznacza braku wszystkich surowych źródeł |
| `configs/polymarket_btc5m_fee_regimes_v2.json` | Lokalna zamrożona konfiguracja opłat istnieje | Event-level historyczne fee niezweryfikowane; D0019 fallback `$5 USDC` pozostaje |

Kompletna lista fizycznych ścieżek, rozmiarów i dowodów znajduje się w [CSV inwentarza](btc_D_holdout_local_source_inventory_20261029.csv) oraz [JSON](btc_D_holdout_local_source_inventory_20261029.json). [Maszynowa lista luk](btc_D_holdout_local_source_gaps_20261029.csv) rozpisuje każdą brakującą godzinę PMXT osobno.

## Luka księgi, cutoffs i kwalifikacja źródeł

Manifest archiwum wskazuje brak pełnego venue od `2026-08-15T10:00:00Z` do wyłącznego `2026-08-18T06:00:00Z`. Skan samych nazw plików potwierdza 68 brakujących godzin. Jest to holdoutowy brak event tape i należy go odróżnić od 24 historycznych godzin T0010 development: `[2026-04-13T19:00Z, 2026-04-14T16:00Z)` oraz `[2026-06-11T04:00Z, 2026-06-11T07:00Z)`.

Według dozwolonych flag PMXT `archive_history_complete`:

- 27 107 z 28 222 rynków ma complete entry-history window metadata.
- 1 115 rynków ma niekompletne okno: 816 nie ma entry-hour/freshness-window metadata, a 299 ma entry-hour, lecz nadal brakuje części 25/26-hour historii.
- 816 startów bez entry-hour obejmuje `2026-08-15T10:05Z`–`2026-08-18T06:00Z` włącznie; źródłowa etykieta to `full_venue_gap` dla 815 oraz `v3` dla granicznego startu `2026-08-18T06:00Z`.
- Pozostałe 299 startów z niekompletną historią, ale z entry-hour, obejmuje `2026-08-18T06:05Z`–`2026-08-19T06:55Z` włącznie.

Z samej obecności indeksu/plików nie wynika prawidłowy BBO, świeżość, pełna głębokość, bezbłędna kolejność eventów ani możliwość fillu. Wymagana przyszła kwalifikacja zachowuje symetryczny source-quality mask E0038 dla T−60/T−59 oraz nie zamyka FAILED gate parytetu E0037. D0019 nadal zabrania wstecznego stosowania bieżącego pola `order_min_size_shares_current` jako historycznego minimum USDC; bez zweryfikowanego historycznego minimum używa się zamrożonego fallbacku `$5 USDC`, bez zaokrąglania w górę. E0040 pozostaje źródłem istniejących ścieżek rozliczeń terminalnych, nie dowodem na holdout.

`T0046`/`T0048`: sprawdzona konfiguracja E0039/E0040 ma `T0046_data_or_process_changes=false`; wystąpienia T0048 są historycznymi zapisami/procesem aktywnym przy wcześniejszym gate. Nie znaleziono zależności wymagającej ich zmiany dla tej inwentaryzacji ani zakresu gap fill. Zadania/procesy zachowano i nie ingerowano w nie.

## Plan następnego autoryzowanego etapu

Nie uruchamiano budowy wejść. Po osobnej autoryzacji przygotowania wejścia, a dopiero po uzupełnieniu/rozstrzygnięciu źródeł:

1. Odczytać z E0041 developmentowego Parquet tylko zamrożone 70 pól BASE i `Opened` strumieniowo; odtworzyć deterministyczny stan memory/ESN od kanonicznego resetu. Sidecar nie ma stanu, więc zapisany `dynamic_features.parquet` nie wystarcza do kontynuacji. Użyć niezmienionych transformów z `btc_candidate_quality_freeze_20261022/bundles/transforms/` oraz istniejących `_transform_names`, `_build_dynamic_feature_artifact`, `features/causal_memory_control.py` i `features/esn_reservoir.py`; nie fitować modeli ponownie.
2. Dla holdoutu wziąć tylko 64 potrzebne BASE pola z szerokiego 656-kolumnowego feature datasetu oraz sześć Z/V z `holdout_flow_features.parquet`, połączyć po `Opened`. Każdy wymagany wiersz spełniać ma `Opened = market_start_utc - 2 min`; pierwszy holdout wiersz `Opened=2026-06-30T23:58Z`. Zachować pierwszą decyzję `23:59Z` oraz T−59 execution cutoff.
3. Użyć Gamma `(condition_id, market_start_utc, up_token_id, down_token_id)` i PMXT technical/event archive jako adapterów native book. Po źródłowym uzupełnieniu rekonstruować obie strony chronologicznie, zachować initialize event przed cutoffem oraz tylko zdarzenia z receive time `<= market_start−60s` i `<= market_start−59s` odpowiednio. Fee derive at both cutoffs. Zastosować E0038 symmetric qualification; nie uznawać indeksu za kontrolę poprawności event stream.
4. Zastosować E0040 settlement path i D0019 minimum-unit fallback; official Gamma outcome remains settlement source. Nie mapować slotów bez rozstrzygnięcia na zero-PnL/no-trade.
5. Dopiero zatwierdzony source-only builder może zapisać `data/analysis/polymarket/BTC/btc_D_holdout_evaluation_20261028/evaluation_rows.parquet` i `evaluation_input_manifest.json`. Builder/materializer nie istnieje obecnie. Chroniony evaluator `run_btc_D_holdout_evaluation_20261028.py` wymaga osobnej `evaluation_authorized=true`; nie uruchamiać go w etapie budowy/inwentaryzacji.

Punkty wejścia istniejące dziś: `run_btc_chronology_reconstruction_20261027.py` (`_transform_names`, `_build_dynamic_feature_artifact`, `_load_fold_arrays`), `features/causal_memory_control.py`, `features/esn_reservoir.py`, `run_btc_t0009_resolution_coverage_20261026.py` (`_replay_development_sources`), `utils/btc_t0009_source_audit.py` (`source_quarantine_reasons`, `source_quality_mask`, `replay_native_book`) oraz evaluator E0041. Brakuje holdoutowego state continuation/materializera, przeniesienia source-quality adaptera na ten pełny holdout, luki źródła PMXT i końcowego manifest buildera. E0037/E0038/E0040 gates remain unchanged.

## Objętość i zasoby (estymata, nie pomiar)

- Fizyczne źródła główne: E0041 development feature Parquet 5.383 GB, szerszy feature Parquet 4.757 GB, PMXT hourly event parts 5.296 GB, holdout flow Parquet 30.7 MB, 98 surowych JSON 18.7 MB oraz indeksy Gamma/PMXT razem ok. 18.2 MB. OHLCV archiwa/CSV i alternatywne book źródła są dodatkowo wymienione w inventory CSV. Rozmiar pliku to górna granica odczytu pełnego; przy projekcji kolumn faktyczny I/O będzie niższy, ale nie został zmierzony.
- Skala indeksu: 4 137 istniejących godzin PMXT; 68 brakujących holdout godzin; 28 222 potwierdzone starty; 184 320 flow minute rows.
- Docelowe tablice 86 wartości float64 × 28 222 ≈ 19.4 MB przed kompresją. Na podstawie proporcji T0009 dev snapshotów para T−60/T−59 to około 13.1 MB, coverage manifest około 3 MB. Wstępny zakres dla `evaluation_rows.parquet` + manifest to 25–50 MB.
- Przy strumieniowaniu, checkpointach godzinowych i bez kopiowania całych 5+ GB Parquet, dodatkowy szczyt staging/temp jest szacowany na mniej niż 200 MB ponad biblioteki i bufory; nie mierzono RAM ani rzeczywistego temporary peak. Wznawianie po godzinie/partycji powinno zachować oddzielne pary cutoffów i identity/hashes.

## Przeszukane lokalizacje

- `data/analysis/polymarket/BTC/btc_candidate_holdout_20261022/source_completion/` (raw_by_day, flow output, manifesty).
- `data/analysis/polymarket/BTC/btc_base70_raw_holdout_20261018/` (holdout flow, daily archive staging; nie otwierano sibling predictions/economics).
- `data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/` (szeroki dataset cech).
- `data/analysis/polymarket/BTC/btc_D_holdout_preparation_20261028/` (E0041 bundle, development inputs, dynamic files).
- `data/analysis/polymarket/BTC/preopen_v1/t59_reassessment_20261007/` oraz `.../t59_reassessment_20261008/pmxt_extended/` (Gamma/PMXT indices, manifest, pliki partycji jako nazwy/rozmiary).
- `data/analysis/polymarket/BTC/btc_t0009_resolution_coverage_20261026_38a6221f0d90/pmxt_replay/` (T0009 development snapshots).
- `data/analysis/polymarket/BTC/btc_preopen_taker_flow_20261018/` (June ZIP i lista memberów).
- `data/analysis/btc_chronology_reconstruction_20261027/raw_prefixes/` i `flow_prefixes/` (istniejące developmentowe prefiksy/konfiguracje).
- `data/raw/polymarket/obadiaha/` i `data/raw/polymarket/kachoio/` (alternatywne, wcześniejsze archiwa).
- `data/datasets/_tmp/binance_vision/um/klines/BTCUSDT1m/` (wskazane w manifeście CSV, tylko stat fizycznej obecności/rozmiaru).
- `configs/` i `utils/` dla fee/minimum i istniejących adapterów.

Nie przeglądano prywatnych katalogów ani innych projektów; nie wykonano szerokiego przeglądu dostawców ani pobierania.

## Proceduralna uwaga

Podczas wcześniejszego przeglądu otwarto zbyt szeroko istniejący raport holdoutu zawierający zagregowane historyczne metryki predykcyjne. Odczytano również najwyższy poziom mieszanego `btc_base70_raw_holdout_20261018_manifest.json` jako kontekst struktury; nie drukowano ani nie wykorzystano jego wartości. Żadna z tych historycznych metryk ani pól mieszanych nie została skopiowana, porównana lub użyta do tej inwentaryzacji. Ta praca nie odczytała surowych holdout prediction/outcome/economic payload rows, nie wygenerowała predykcji/metryk/ekonomiki i nie zmieniła `evaluation_authorized`.

## Wynik i rejestry

T0014 jest zakończoną inwentaryzacją metadata-only; E0042 zapisuje te ustalenia. E0037 pozostaje FAILED, T0009 BLOCKED, T0010 DEFERRED, D0019 aktywne, E0041 `D_BUNDLE_READY`; T0046/T0048 zachowane. `evaluation_authorized=false`, `activated=false`, `HOLDOUT_NOT_RUN` pozostają bez zmian. Poprzednia tabela 2026-10-28 jest zachowana w historii Git; jej nieaktualne 28 220 jest tu jawnie zastąpione przez indeksowe 28 222 i dwa jawne unmatched slots.
