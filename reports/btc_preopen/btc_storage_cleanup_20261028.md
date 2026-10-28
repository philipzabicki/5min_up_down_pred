# Lokalna kompresja i sprzątanie danych — 2026-10-28

Powiązane zadanie: T0013. Operacje wykonano lokalnie na woluminie D:. Nie przesyłano danych do usług zewnętrznych. Nie tworzono archiwów ani konwersji tabel; usunięto dwa ukończone cache E0039 oraz jeden potwierdzony, niekompletny plik tymczasowy.

## Wynik pomiaru

Wolne miejsce zmierzone na woluminie wzrosło o **13 427 347 456 B (12,505192 GiB)** w trzech operacjach. Ostatni odczyt po uruchomieniu i zakończeniu próby E0041 v2 wyniósł **226 158 792 704 B (210,626789 GiB) wolnego**.

| Pomiar | Wolne bajty | GiB |
|---|---:|---:|
| Początek audytu | 82 075 066 368 | 76,438362 |
| Przed usunięciem cache F1/F2 | 213 191 921 664 | 198,550449 |
| Po usunięciu cache F1/F2 | 224 289 546 240 | 208,885918 |
| Przed usunięciem pliku tymczasowego | 223 829 098 496 | 208,457092 |
| Bezpośrednio po usunięciu pliku tymczasowego | 226 158 821 376 | 210,626816 |
| Ostatni odczyt po zakończeniu runnera | 226 158 792 704 | 210,626789 |

Między pierwszym pomiarem a pierwszą operacją wolne miejsce wzrosło dodatkowo o 131 116 855 296 B (122,112087 GiB); źródła tej równoległej zmiany nie ustalono i nie zaliczono jej do odzysku z tego zadania. Podczas pracy E0041 między dwiema operacjami wolne miejsce spadło o 460 447 744 B; w tym czasie runner zapisał m.in. `dynamic_features.parquet`, OOF i modele. Nie przypisuję całej różnicy konkretnym zapisom bez pełnej ewidencji alokacji systemu plików.

## Wykonane operacje

| Ścieżka | Rola i dowód | Rozmiar / SHA-256 | Wynik i odzysk |
|---|---|---:|---|
| `data/analysis/polymarket/BTC/btc_chronology_reconstruction_20261027/runs/508858c48b11b925/F1_20260415_20260501/20914cfbc01dd0b3/selection_pipeline/441e9566cadfbf70/stages/causal_candidate_feature_join_ec93eb01fd79/feature_dataset_with_causal_candidates.parquet` | Ukończony pośredni join cech dla development F1. Fold zakończył się 2026-10-27; sidecar podaje zakres `2020-06-09`–`2026-04-30`, 3 098 307 wierszy i 662 kolumny. Zachowano źródłowy flow prefix oraz końcową macierz folda. Jeden hardlink. | 5 541 884 234 B; `903f6425188dcb4104b5afe8e87f3659f058215711d72f8f0c22159b3fb8b1ba` | Usunięto po pełnym SHA-256 i kontroli zależności. |
| `data/analysis/polymarket/BTC/btc_chronology_reconstruction_20261027/runs/508858c48b11b925/F2_20260501_20260601/d1ec3319b9294e29/selection_pipeline/4cc33bc2eb314027/stages/causal_candidate_feature_join_439298c0ce05/feature_dataset_with_causal_candidates.parquet` | Ukończony pośredni join cech dla development F2. Fold zakończył się 2026-10-27; sidecar podaje zakres `2020-06-09`–`2026-05-31`, 3 142 947 wierszy i 662 kolumny. Zachowano źródłowy flow prefix oraz końcową macierz folda. Jeden hardlink. | 5 555 736 541 B; `7baa06b0366ac19c96bcfcbd2906157e5bbedffef88968a5cc6119a1ae478653` | Usunięto po pełnym SHA-256 i kontroli zależności. |
| `data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/shared_runs/calibration_seed/9ca4755fd7b45c9f/stages/final_feature_dataset_4527ce62c44f/dataset/.BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.28244.tmp.parquet` | Niekompletny scratch z historycznego etapu. Nie występuje w `feature_dataset_stage.json`; odczyt stopki zgłosił brak poprawnej stopki Parquet. Obok zachowano ukończony plik wynikowy i metadane etapu: 3 327 267 wierszy; jego stopka jest poprawna. Jeden hardlink. Brak procesu historycznego runu. | 2 329 720 393 B; `b869fb5bf606fa354aeba35a9901bf7703e73d3a6b51c80214809fd7919d8d8d` | Usunięto po kontroli metadanych, poprawności zachowanego wyniku, SHA-256 i liczby hardlinków. |

Usunięcie pierwszej pary zwiększyło zmierzoną wolną przestrzeń o 11 097 624 576 B, czyli o 3 801 B więcej niż suma długości logicznych tych plików. Usunięcie pliku tymczasowego zwiększyło ją o 2 329 722 880 B, o 2 487 B więcej niż długość logiczna. Łączna różnica 6 288 B wynika z alokacji systemu plików i pomiaru wolnego miejsca. Hashowanie pierwszej pary trwało łącznie 9,273 s, ich usunięcie 1,023 s. Hashowanie pliku tymczasowego trwało 1,970 s, usunięcie 0,102 s. Współczynnik kompresji nie dotyczy usuniętych plików.

## Odtworzenie i archiwa

Nie utworzono archiwów. Brakujące macierze join F1/F2 można odtworzyć z zachowanych macierzy bazowych i flow prefixów, uruchamiając zamrożony runner E0039 bez argumentów:

```powershell
python run_btc_chronology_reconstruction_20261027.py
```

Kod `_run_stage` ponownie wykonuje etap, gdy zarejestrowany artefakt zniknął lub nie pasuje do manifestu. Runner przechodzi własną bramkę zasobów i może ponownie użyć pozostałych zweryfikowanych etapów. Nie uruchamiałem go podczas tego sprzątania. Usunięty plik tymczasowy nie wymaga odtworzenia: poprawny, ukończony dataset i jego metadane pozostały na miejscu.

## Zachowane dane i nierozstrzygnięci kandydaci

- Cały aktywny run E0041 `189049366841f6d025e3d9b080bb69dd0e441bb5638e451708f27c0635754c44`, jego wejścia, bezpieczny raw prefix, study, budżety, transformacje, dataset, OOF i dziewięć zapisanych modeli D pozostawiono. T0046/T0048 i ich procesów nie zatrzymywano ani nie modyfikowano.
- Zachowano końcowe macierze i predykcje E0039/E0040, kalibratory, decyzje, ledgery, manifesty, raporty, raportowane hashy oraz źródła. Zachowano też końcowe macierze F1/F2 E0039, z których można odbudować usunięte joiny.
- F3 E0039 i stare retry zawierające materiały o możliwym mieszanym zakresie holdoutu pozostają nietknięte. Nie odczytywano ich payloadu do kwalifikacji.
- Duże tablice ESN z ukończonych badań, macierz F1/F2 po joinie z ukończonej ścieżki oraz inne historyczne cache pozostają kandydatami do osobnego audytu zależności. Nie usunięto ich, bo ich rola w dalszym odtwarzaniu nie została tu wykazana.
- Pozostały dwa małe pliki `.tmp` (328 316 B i 2 265 B) poza aktywnym E0041; nie stanowiły istotnego odzysku i nie były objęte listą operacji.

## Wznowienie D

Dokładne polecenie `python run_btc_D_holdout_preparation_20261028.py` przeszło niezmieniony gate o `2026-10-28T21:52:21Z`: 208,89 GiB wolnego, 47,77 GiB dostępnego RAM i 6 350 MiB wolnej pamięci GPU. Wznowiło ten sam run. Bieżąca próba trwała 265,436 s; skumulowany licznik całego przygotowania osiągnął 13 779,146 s.

Runner wytworzył 78 622 wiersze development calibration OOF, dopasował Platt na 21 951 oficjalnych etykietach dostępnych do `2026-06-30T23:59:00Z` i zapisał wszystkie dziewięć modeli D po 86 cech (3 186 139 wierszy treningowych na członka). Hashy dziewięciu plików modeli zgodnych z sidecarami sprawdzono. Kalibrator ma ostatnią datę zwolnienia etykiety `2026-06-30T23:56:22Z`.

Przy pakowaniu metadanych runner zgłosił `KeyError: 'rows'` w `run_btc_D_holdout_preparation_20261028.py:1291`. `fit_official_platt` zapisał obiekt z polem `training_rows`, a pakujący kod oczekiwał pól `rows` i `required`. Nie powstał `bundle_manifest.json`, a `preparation_manifest.json` zachowuje `interrupted_or_failed_with_checkpoints`. Nie ponawiałem zamrożonego runnera. `evaluation_authorized=false`; odczytane i zmaterializowane wiersze holdoutu oraz predykcje holdoutu: zero. E0037 pozostaje FAILED, T0009 BLOCKED. Szczegóły w [rejestrze E0041](btc_D_holdout_preparation_20261028.md).

## Weryfikacja

- SHA-256, rozmiary i tożsamość hardlinków sprawdzono przed usunięciem; pliki docelowe po usunięciu nie istnieją.
- Dla częściowego pliku sprawdzono stopkę Parquet, sidecar ukończonego etapu i zachowany wynik docelowy.
- E0041 pozostaje bez bundle manifestu, z wyłączoną autoryzacją i zerowym odczytem/predykcjami holdoutu.
- Testy modelu ani pełny backtest nie uruchamiano. Kod i protokoły nie zmieniono.
