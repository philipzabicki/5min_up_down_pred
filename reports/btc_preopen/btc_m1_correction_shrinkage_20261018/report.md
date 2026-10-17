# M1 korekta skurczona jednym parametrem — 2026-10-18

> M1 był wybrany wcześniej z udziałem maja i czerwca. Chronologiczne dopasowanie α w maju nie czyni czerwca niezależnym testem całej procedury.

## Protokół i zamrożenie

Końcowy manifest poprawionego przebiegu zapisano przed ponowną oceną ekonomiczną czerwca: `2026-10-17T22:50:46.609083+00:00` (SHA-256 `941d49973a72794d9d675c027f8472edc046dad1836cbea57a0ffcd47ab86ce1`). Pierwszy manifest z tym samym α został zamrożony przed pierwszym czerwcowym scoringiem.
Użyto zapisanych foldowych, przyczynowych korekt surowego logitu M1 z konfiguracji `leaves4_depth2_l2_100`. Nie trenowano modelu, nie odtwarzano holdoutu, nie przeliczano cech ani OOF.
Odtworzenie `alpha=0` dało rynek z maksymalnym błędem 0.000000000000000; `alpha=1` odtworzyło zapisane p_M1 dokładnie (maksymalny błąd 0.000000000000000). Użyto surowych korekt, bez utraty precyzji przy ich odtwarzaniu.

Majowe dopasowanie zawiera wszystkie outcome-quality wiersze z market start w maju i etykietą dostępną do 2026-06-01T00:00:00Z; nie filtruje po EV, transakcji ani wycenialności.
Wiersze maja: 5,253; kwalifikujące się do dopasowania: 5,252; wyłączone przez dostępność etykiety: 1. Reguła: `max(resolved_at_utc, market_start_utc + 5 minutes) + 60 seconds`.

## Dopasowanie α w maju (wynik dopasowania)

Wybrane α = **0.074270374954** (`interior_derivative_root_bisection`), w ograniczeniu [0, 1]. Minimalizowano nieważony log-loss na 5,252 wierszach.

| α | Log-loss celu | Pochodna celu |
|---:|---:|---:|
| 0 (rynek) | 0.692822099 | -0.000224578046 |
| 0.074270374954 (wybrane) | 0.692813759 | -0.000000000000 |
| 1 (M1) | 0.694107525 | 0.002791469851 |

Maj jest próbką dopasowania, a nie późniejszą walidacją. Bazową konfigurację M1 wybrano wcześniej z użyciem maja i czerwca; czerwiec ocenia zamrożone α warunkowo na tym wybranym M1.

## Jakość prawdopodobieństw w czerwcu

Wszystkie trzy warianty liczone są na tych samych czerwcowych OOF wierszach. Ujemna różnica kalibracja minus porównanie oznacza poprawę.

| Wariant | n | Log-loss | Brier | Śr. bezwzględna korekta p względem rynku |
|---|---:|---:|---:|---:|
| market | 3,422 | 0.693243840 | 0.250043315 | 0.000000000 |
| original_M1 | 3,422 | 0.692347593 | 0.249595301 | 0.016675462 |
| calibrated_M1 | 3,422 | 0.693120070 | 0.249981496 | 0.001239149 |

| Sparowana różnica czerwiec | Δ log-loss | 95% CI, blok 3 dni UTC | Δ Brier | 95% CI, blok 3 dni UTC |
|---|---:|---:|---:|---:|
| kalibrowany M1 − oryginalny M1 | 0.000772476 | [-0.001140322, 0.002279112] | 0.000386195 | [-0.000567589, 0.001136564] |
| kalibrowany M1 − rynek | -0.000123770 | [-0.000244300, 0.000029576] | -0.000061819 | [-0.000122045, 0.000014903] |

Bootstrap: 2,000 poprawnych replikacji, seed 20261018; przedziały warunkowe na wybranym M1 i zamrożonym α. Nie obejmują pełnej niepewności wcześniejszego wyboru modelu.

## Ekonomia czerwcowa, stałe $5

Decyzja T−60, wykonanie T−59, obie strony wyceniane ponownie dla każdego wariantu, próg EV netto 1% stawki, istniejące opłaty, głębokość, fills i cutoff rozliczeń 2026-07-01T00:00:00Z. Predykcyjne EV i wynik rzeczywisty w dekompozycji odnoszą się do tych samych rozliczonych wykonań EV ≥ 1%.

| Wariant | Wycenialne rynki | Dodatnie EV (rynki / strony) | Decyzje EV ≥1% | Wypełnienia T−59 | Rozliczone | Obrót | Opłaty | PnL netto | PnL/obrót | Σ EV T−60 na tych wykonaniach |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| market_alpha_0 | 3,143 | 0 / 0 | 0 | 0 | 0 | $0.00 | $0.0000 | brak transakcji | — | brak transakcji |
| original_M1_alpha_1 | 3,143 | 851 / 851 | 801 | 792 | 792 | $3960.00 | $138.5985 | $-38.2725 | -0.9665% | $77.2068 |
| calibrated_M1 | 3,143 | 0 / 0 | 0 | 0 | 0 | $0.00 | $0.0000 | brak transakcji | — | brak transakcji |

EV ≥1% — agregaty na wspólnym dla każdej polityki zbiorze jej rozliczonych wykonań:

| Wariant | Σ E60 | Σ E59 | Σ R59 | R59 − E60 | E59 − E60 | (y−p) × net shares | Maks. błąd tożsamości |
|---|---:|---:|---:|---:|---:|---:|---:|
| market_alpha_0 | brak transakcji | brak transakcji | brak transakcji | brak transakcji | brak transakcji | brak transakcji | — |
| original_M1_alpha_1 | $77.2068 | $76.6684 | $-38.2725 | $-115.4793 | $-0.5383 | $-114.9410 | 0.000000000000 |
| calibrated_M1 | brak transakcji | brak transakcji | brak transakcji | brak transakcji | brak transakcji | brak transakcji | — |

Dla α=1 luka między sumą przewidywanego EV T−60 a rzeczywistym wynikiem wyniosła $-115.4793 w EV≥1%. Dla skalibrowanego M1 nie było rozliczonych wykonań w tej grupie, więc nie ma podstaw do wniosku, że luka ekonomiczna się zmniejszyła.

## Uwaga o pierwszym wykonaniu

Pierwszy przebieg ekonomiczny nie kwalifikował żadnej strony, ponieważ dziennik predykcji przekazany do silnika nie zawierał token IDs UP/DOWN. Jego manifest, wyniki i artefakty zachowano w `data/analysis/polymarket/BTC/btc_m1_correction_shrinkage_20261018/prior_invalid_execution`. Jego czerwcowe metryki jakości były poprawne i odpowiadają tym odtworzonym tutaj; wyłącznie ekonomię oznaczono jako nieważną. Poprawiony przebieg użył tego samego α i prawdopodobieństw, a jego manifest z hashami poprawionego kodu zapisano przed ponowną oceną ekonomiczną.


Wartości z pustych grup EV są prezentowane jako „brak transakcji”, a nie jako zerowa rozbieżność EV–wynik. Wcześniejszy poprawny przebieg obliczeniowy, z tą różnicą prezentacji, zachowano w `data/analysis/polymarket/BTC/btc_m1_correction_shrinkage_20261018/prior_valid_preformat`.

Kalibracja zmieniła wybrany kierunek względem M1 w 851 z 3,422 czerwcowych rynków. Źródłem nowej polityki są zapisane snapshoty T−60 i T−59 obu stron; nie użyto wcześniejszej tabeli diagnostycznej zawierającej tylko poprzednio wybraną stronę.

Depth, opłaty, wynik i obrót w podziale na pełny order book i proxy T−59:

| Wariant | Głębokość | Rozliczone fills | Obrót | Opłaty | PnL netto |
|---|---|---:|---:|---:|---:|
| market_alpha_0 | brak transakcji | 0 | $0.00 | $0.0000 | $0.0000 |
| original_M1_alpha_1 | full_book | 792 | $3960.00 | $138.5985 | $-38.2725 |
| calibrated_M1 | brak transakcji | 0 | $0.00 | $0.0000 | $0.0000 |

## Artefakty i wykonanie

- Predykcje wyłącznie z development: `data/analysis/polymarket/BTC/btc_m1_correction_shrinkage_20261018/development_predictions.parquet` (SHA-256 `697afdfebe9e3a82c49acf14e665e2ae731a96eb1bb54d936f2d660be2502528`).
- Manifest przed raportowanym przebiegiem czerwcowym: `data/analysis/polymarket/BTC/btc_m1_correction_shrinkage_20261018/frozen_manifest.json`.
- Pełny dziennik decyzji i filli czerwcowych: `data/analysis/polymarket/BTC/btc_m1_correction_shrinkage_20261018/june_economic_decisions.parquet`.
- Czas całego runnera: 5.380 s; szczytowy working set procesu: 1.699 GiB.
- Python 3.14.4, pandas 3.0.2, NumPy 2.4.4.

Reprodukcja bez argumentów CLI: `python run_btc_m1_correction_shrinkage_20261018.py`.
