# BASE70 FULL_FEATURES: feature_fraction=1.0 — 2026-10-18

## Cel i zakres

Hipoteza zamrożona przed fitami: przy słabym sygnale losowe pomijanie części z 70 wybranych cech może obniżać stabilność prognoz istotnych dla kwalifikacji transakcji. To hipoteza do sprawdzenia, nie ustalona przyczyna wcześniejszych wyników.

Główny kontrast to FULL_FEATURES − SINGLE. ENSEMBLE jest dodatkowym kontekstem i pochodzi z istniejących predykcji pięciu komponentów o równych wagach; żaden seed nie został wybrany po wyniku. Nie trenowano ponownie SINGLE ani ENSEMBLE.

## Zamrożone porównanie i źródła

- Macierz: 3,174,805 wierszy i 70 cech w niezmienionej kolejności; target `target_preopen_5m_up_binance_index_proxy`; te same 10 foldów, purging i sample weights. Hash wag: `3a21a8cfecaeb10cdf0af1e63fd9833eef034954f6f530005bab71da6f1397ca`.
- Dataset i indeksy treningowe/walidacyjne zostały zweryfikowane względem istniejących źródeł. Macierz wczytano raz; cechy, profile VP/RP, wskaźniki ani selekcja nie były generowane ponownie.
- Parametry budowania Dataset: kontrola używała `lgb.Dataset(X, label, weight, feature_name, free_raw_data=False)` przekazanego do `lgb.cv`; FULL_FEATURES zastosował challenger params przez `_update_params()` przed `construct()` i `Dataset.subset(sorted(train_idx))` na zapisanych foldach. Seed próbkowania histogramów pozostał `data_random_seed=20261015`.
- Oryginalny OOF SINGLE, jego 10 boosterów oraz zapisany OOF ENSEMBLE zweryfikowano i wykorzystano bez refitu. Hash bundle OOF SINGLE/ENSEMBLE: `e32aab4d2a9094e12c75df63d7d333a32f7a4d0be15eb79190ddc9853de436d9`.
- Macierz BASE70 SHA-256: `06952bbf1b2b0a4e9fc4cc918490fbadc540c897815b8105242201c6148dfe19`; foldy SHA-256: `02768c3ddd80e26720507757e065337bc54afd22691bf830c021698c49133c07`; OOF SINGLE SHA-256: `dfaad30e79c2c98333f02f9a6da3e99686c7d5a3df1ded5a80a0d01bd98a99e8`.
- Kalendarz ekonomiczny 20 735 decyzji, snapshoty T−60/T−59, etykiety oficjalne i ledger źródłowy zostały zweryfikowane względem hashy istniejących manifestów.

### Efektywne parametry

| Parametr | SINGLE | FULL_FEATURES |
|---|---:|---:|
| `bagging_seed` | `20261015` | `20261015` |
| `boosting_type` | `gbdt` | `gbdt` |
| `data_random_seed` | `20261015` | `20261015` |
| `device_type` | `gpu` | `gpu` |
| `feature_fraction` | `0.7962744599393133` | `1.0` |
| `feature_fraction_seed` | `20261015` | `20261015` |
| `feature_pre_filter` | `False` | `False` |
| `gpu_use_dp` | `False` | `False` |
| `lambda_l1` | `9.508616956284627` | `9.508616956284627` |
| `lambda_l2` | `30.729843292781943` | `30.729843292781943` |
| `learning_rate` | `0.06418546619439842` | `0.06418546619439842` |
| `max_bin` | `63` | `63` |
| `metric` | `None` | `None` |
| `min_data_in_leaf` | `12089` | `12089` |
| `num_leaves` | `81` | `81` |
| `num_threads` | `8` | `8` |
| `objective` | `binary` | `binary` |
| `seed` | `20261015` | `20261015` |
| `verbosity` | `-1` | `-1` |

- Oryginalne seedy zostały odczytane z zapisanego CV: `seed=20261015`, `feature_fraction_seed=20261015`, `data_random_seed=20261015`, `bagging_seed=20261015`. Dataset histogramów użył tego samego `data_random_seed` i tego samego wzorca konstrukcji co kontrolne CV.
- `feature_fraction_bynode` nie jest jawnie ustawione w parametrach kontroli; efektywny default LightGBM 4.6.0 wynosi `1.0`, zatem nie ma dodatkowego losowania cech per node. Nie zmieniono go w challengerze. [Dokumentacja parametrów LightGBM 4.6.0](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#feature_fraction_bynode) opisuje losowanie na poziomie węzła tylko dla wartości poniżej 1.
- Bagging pozostaje wyłączony (`bagging_fraction=1.0`, `bagging_freq=0`). Nie zastosowano early stoppingu, Optuny ani kalibracji. Oba warianty mają dokładnie 31 iteracji.
- `feature_fraction=1.0` nie gwarantuje pełnej deterministyczności GPU ani nie usuwa innych źródeł wrażliwości. LightGBM opisuje GPU jako używające domyślnie sumowania FP32; opcja `deterministic` dotyczy CPU. [Dokumentacja LightGBM 4.6.0](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#core-parameters).

## Jakość probabilistyczna

### Wspólne proxy OOF

| Wariant | Wiersze | Log-loss | Brier | AUC |
|---|---:|---:|---:|---:|
| SINGLE | 318,560 | 0.692134 | 0.249494 | 0.52508 |
| FULL_FEATURES | 318,560 | 0.692154 | 0.249504 | 0.52495 |
| ENSEMBLE | 318,560 | 0.692097 | 0.249475 | 0.52547 |

Δ log-loss/Brier poniżej zera oznacza poprawę FULL_FEATURES; Δ AUC powyżej zera oznacza poprawę.

| Fold | Wiersze | Δ log-loss | Δ Brier | Δ AUC |
|---:|---:|---:|---:|---:|
| 0 | 31,858 | 0.000158 | 0.000079 | -0.001569 |
| 1 | 31,841 | -0.000019 | -0.000009 | -0.000182 |
| 2 | 31,862 | 0.000164 | 0.000082 | -0.001368 |
| 3 | 31,861 | -0.000020 | -0.000011 | 0.001156 |
| 4 | 31,842 | -0.000117 | -0.000058 | 0.001466 |
| 5 | 31,856 | -0.000136 | -0.000068 | 0.001760 |
| 6 | 31,861 | 0.000035 | 0.000018 | -0.000810 |
| 7 | 31,858 | 0.000046 | 0.000023 | -0.000741 |
| 8 | 31,860 | 0.000026 | 0.000013 | -0.000582 |
| 9 | 31,861 | 0.000056 | 0.000028 | -0.001247 |

Bootstrap sparowany: 2 000 replikacji, kołowe bloki UTC po 3 dni, seed `20261018`.

| Miara | Punkt FULL_FEATURES − SINGLE | 95% przedział | Replikacje |
|---|---:|---:|---:|
| log_loss | 0.000019 | -0.000036 do 0.000073 | 2,000 |
| brier | 0.000010 | -0.000018 do 0.000036 | 2,000 |
| auc | -0.000131 | -0.000774 do 0.000551 | 2,000 |

### Oficjalne etykiety

Użyto tych samych 20 718 etykiet rozwojowych; bez holdoutu i bez kalibracji.

| Wariant | Wiersze | Log-loss | Brier | AUC |
|---|---:|---:|---:|---:|
| SINGLE | 20,718 | 0.692052 | 0.249453 | 0.52744 |
| FULL_FEATURES | 20,718 | 0.692049 | 0.249451 | 0.52781 |
| ENSEMBLE | 20,718 | 0.692154 | 0.249503 | 0.52648 |

| Miesiąc UTC | Wariant | Wiersze | Log-loss | Brier | AUC |
|---|---|---:|---:|---:|---:|
| 2026-04 | SINGLE | 3,168 | 0.691610 | 0.249233 | 0.53249 |
| 2026-04 | FULL_FEATURES | 3,168 | 0.691410 | 0.249133 | 0.53456 |
| 2026-04 | ENSEMBLE | 3,168 | 0.691726 | 0.249290 | 0.53085 |
| 2026-05 | SINGLE | 8,917 | 0.692259 | 0.249556 | 0.52463 |
| 2026-05 | FULL_FEATURES | 8,917 | 0.692138 | 0.249496 | 0.52629 |
| 2026-05 | ENSEMBLE | 8,917 | 0.692325 | 0.249589 | 0.52319 |
| 2026-06 | SINGLE | 8,633 | 0.692000 | 0.249427 | 0.52964 |
| 2026-06 | FULL_FEATURES | 8,633 | 0.692192 | 0.249522 | 0.52770 |
| 2026-06 | ENSEMBLE | 8,633 | 0.692133 | 0.249493 | 0.52894 |

Sparowany bootstrap oficjalnych etykiet używa tych samych 2 000 replikacji i bloków 3-dniowych.

| Miara | Punkt FULL_FEATURES − SINGLE | 95% przedział | Replikacje |
|---|---:|---:|---:|
| log_loss | -0.000003 | -0.000239 do 0.000228 | 2,000 |
| brier | -0.000001 | -0.000120 do 0.000114 | 2,000 |
| auc | 0.000370 | -0.003364 do 0.004076 | 2,000 |

Rozkład różnicy prawdopodobieństw `p_FULL_FEATURES − p_SINGLE`:

| Próbka | N | Średnia | SD | Średnia bezwzględna | Udział dodatnich | p01 | p05 | Mediana | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Proxy OOF | 318,560 | 0.000103 | 0.007537 | 0.005999 | 50.57% | -0.017513 | -0.012329 | 0.000104 | 0.012494 | 0.017704 |
| Oficjalna | 20,718 | 0.000072 | 0.007815 | 0.006189 | 50.46% | -0.018629 | -0.012769 | 0.000084 | 0.013038 | 0.018440 |

## Ekonomia po kosztach

Wspólny kalendarz obejmuje 20,735 decyzji. Każdy wariant samodzielnie wybiera stronę i kwalifikację z progiem net EV 1%, niezależnym zleceniem $5, decyzją T−60 i wykonaniem T−59. Snapshoty, opłaty, głębokość, cutoff i księgowanie pozostały bez zmian.

Kontrola odtworzyła historyczny ledger SINGLE: 1,038 rozliczonych wykonań, $5,190.00 obrotu i $267.44 PnL. Zgodność z oryginalnym ledgerem: zaliczona.

| Wariant | Wypełnienia | Obrót | Opłaty | PnL netto | PnL/obrót | Prognozowane EV | PnL − EV |
|---|---:|---:|---:|---:|---:|---:|---:|
| SINGLE | 1,038 | $5,190.00 | $188.84 | $267.44 | 5.15% | $279.60 | $-12.17 |
| FULL_FEATURES | 1,072 | $5,360.00 | $194.48 | $243.44 | 4.54% | $284.09 | $-40.65 |
| ENSEMBLE | 960 | $4,800.00 | $175.10 | $130.41 | 2.72% | $265.31 | $-134.91 |

Miesięcznie:

| Miesiąc UTC | Wariant | Wypełnienia | UP / DOWN | Obrót | Opłaty | PnL netto | PnL/obrót | EV | PnL − EV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-04 | SINGLE | 63 | 14 / 49 | $315.00 | $11.45 | $-10.39 | -3.30% | $9.40 | $-19.78 |
| 2026-05 | SINGLE | 447 | 124 / 323 | $2,235.00 | $83.85 | $66.12 | 2.96% | $177.30 | $-111.18 |
| 2026-06 | SINGLE | 528 | 355 / 173 | $2,640.00 | $93.54 | $211.71 | 8.02% | $92.91 | $118.80 |
| 2026-04 | FULL_FEATURES | 68 | 21 / 47 | $340.00 | $12.31 | $42.12 | 12.39% | $10.66 | $31.45 |
| 2026-05 | FULL_FEATURES | 475 | 169 / 306 | $2,375.00 | $88.44 | $-35.81 | -1.51% | $180.31 | $-216.12 |
| 2026-06 | FULL_FEATURES | 529 | 316 / 213 | $2,645.00 | $93.73 | $237.13 | 8.97% | $93.11 | $144.02 |
| 2026-04 | ENSEMBLE | 60 | 17 / 43 | $300.00 | $10.88 | $-25.05 | -8.35% | $9.22 | $-34.27 |
| 2026-05 | ENSEMBLE | 413 | 129 / 284 | $2,065.00 | $77.79 | $-29.58 | -1.43% | $170.73 | $-200.31 |
| 2026-06 | ENSEMBLE | 487 | 323 / 164 | $2,435.00 | $86.42 | $185.04 | 7.60% | $85.36 | $99.68 |

Podział UP/DOWN:

| Wariant | Strona | Wypełnienia | Obrót | Opłaty | PnL | EV | PnL − EV |
|---|---|---:|---:|---:|---:|---:|---:|
| SINGLE | UP | 493 | $2,465.00 | $88.80 | $183.41 | $134.64 | $48.77 |
| SINGLE | DOWN | 545 | $2,725.00 | $100.05 | $84.03 | $144.97 | $-60.94 |
| FULL_FEATURES | UP | 506 | $2,530.00 | $90.94 | $140.60 | $134.37 | $6.23 |
| FULL_FEATURES | DOWN | 566 | $2,830.00 | $103.54 | $102.84 | $149.72 | $-46.88 |
| ENSEMBLE | UP | 469 | $2,345.00 | $84.73 | $151.06 | $128.23 | $22.83 |
| ENSEMBLE | DOWN | 491 | $2,455.00 | $90.37 | $-20.65 | $137.08 | $-157.73 |

Koncentracja PnL w pięciu najlepszych dniach (dni bez transakcji pozostają w kalendarzu bootstrapu):

| Wariant | Pięć najlepszych dni | Udział w PnL | PnL pozostałych dni |
|---|---:|---:|---:|
| SINGLE | $218.21 (2026-06-09: $70.30, 2026-06-03: $54.43, 2026-05-31: $35.05, 2026-05-30: $31.48, 2026-06-07: $26.94) | 81.59% | $49.23 |
| FULL_FEATURES | $197.67 (2026-06-07: $49.14, 2026-06-03: $39.27, 2026-06-09: $37.31, 2026-05-26: $36.45, 2026-06-10: $35.49) | 81.20% | $45.77 |
| ENSEMBLE | $196.43 (2026-06-03: $49.02, 2026-06-07: $44.34, 2026-05-26: $41.21, 2026-06-09: $36.12, 2026-05-30: $25.75) | 150.63% | $-66.03 |

Sparowane 95% przedziały różnic względem SINGLE, 2 000 bootstrapów bloków 3-dniowych, z zachowaniem pełnego kalendarza 72 dni:

| Miara | Punkt FULL_FEATURES − SINGLE | 95% przedział | Replikacje |
|---|---:|---:|---:|
| Średni dzienny PnL | $-0.3334 | $-3.2037 do $2.5537 | 2,000 |
| PnL/obrót | -0.61% | -5.06% do 3.83% | 2,000 |

Decompozycja różnicy rozliczonego PnL; suma sprawdzona: **$-24.00**. Zgodność: `True`.

| Składnik Δ PnL (FULL_FEATURES − SINGLE) | USD | Rynki |
|---|---:|---:|
| Wypełnienia wspólne, ta sama strona | $0.00 | 763 |
| Wypełnienia tylko jednego wariantu | $-24.00 | 584 |
| Wypełnienia wspólne, przeciwne strony | $0.00 | 0 |

## Runtime, zasoby i ograniczenia

- Dziesięć fitów po 31 iteracji zajęło łącznie 15.60s (zakres na fold 1.47–1.73s). Pełny etap jakości trwał 39.41s, a ekonomia i audyty trzech wariantów 48.93s.
- Ostatni przebieg runnera trwał 5.36s i wznowił 10 zgodnych checkpointów; to czas wznowienia z cache, nie łączny czas wcześniejszych obliczeń.
- Szczytowy working set procesu: 5.13 GiB. Środowisko i współbieżne obciążenie zapisano w run manifeście.
- Zapisano 10 boosterów FULL_FEATURES i OOF po każdym foldzie. Runner bez argumentów CLI pomija tylko checkpointy o zgodnej tożsamości, hashu, indeksach walidacyjnych, kolejności cech i liczbie iteracji.
- Eksperyment jest warunkowy względem wcześniej wybranego BASE70, parametrów i development sample. Nie oceniano holdoutu, nie pobierano danych, nie odtwarzano archiwum ani nie wykonywano refitu wdrożeniowego.

## Wniosek

- **Jakość probabilistyczna:** Na proxy OOF FULL_FEATURES zmienił log-loss o 0.000019 (sparowany 95% CI -0.000036 do 0.000073); poprawił log-loss w 4/10 foldów, Brier w 4/10, a AUC w 3/10. Na oficjalnych etykietach różnice wyniosły Δ log-loss -0.000003, Δ Brier -0.000001, Δ AUC 0.000370.
- **Decyzje po kosztach:** FULL_FEATURES wykonał 1,072 zleceń i osiągnął $243.44 PnL wobec 1,038 i $267.44 dla SINGLE. 95% CI dla różnicy średniego dziennego PnL to $-3.20 do $2.55; dla PnL/obrót: -5.06% do 3.83%.
- **Czy wyniki uzasadniają dalszą pracę z FULL_FEATURES:** Nie ma spójnych przedziałowo dowodów poprawy jakości i ekonomii. Wynik nie uzasadnia priorytetu FULL_FEATURES ani refitu; ewentualny dalszy test wymaga osobnej hipotezy i zamrożonego protokołu. Punktowy PnL sam nie rozstrzyga. Ten eksperyment nie wybiera zwycięzcy ani seedu. SINGLE pozostaje historycznym punktem odniesienia, nie potwierdzonym najlepszym modelem. Nie uruchomiono kolejnego eksperymentu.

Runner: `python run_btc_base70_full_features_20261018.py` (plik: `run_btc_base70_full_features_20261018.py`); bez argumentów CLI. Manifest: `reports/btc_preopen/btc_base70_full_features_20261018_manifest.json`.
