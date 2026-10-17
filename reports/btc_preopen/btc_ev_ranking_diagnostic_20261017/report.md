# Diagnostyka rankingu EV BTC — 2026-10-17

> Wyniki M0 i M1 są warunkowe na modelach wybranych z udziałem development OOF. Nie stanowią niezależnego potwierdzenia edge.

## Zakres i zamrożony protokół

Użyto 8,675 zapisanych predykcji development OOF dla każdego z M1, M0 i previous_D, od `2026-05-01T00:05:00+00:00` do `2026-06-30T23:50:00+00:00`. Nie użyto predykcji, etykiet ani wyników holdoutu; ze snapshotów wybrano wyłącznie condition ID z development OOF przed wyceną. Nie trenowano modeli, nie przeliczano BTC/VP/RP/cech ani nie uruchamiano pipeline'u lub replayu archiwum.

Stawka to $5; decyzja T−60, hipotetyczne wykonanie T−59, cutoff rozliczenia `2026-07-01T00:00:00+00:00`. EV i zwrot dzielone są przez żądaną stawkę $5. Przed wyceną kandydatów zapisano [manifest](diagnostic_manifest.json) z hashami, okresami, przedziałami i regułami braków.

Dla każdej pary model–rynek istniejący silnik wycenił UP i DOWN w T−60. Wybrano wyższą wycenialną EV, także gdy była niedodatnia; remisy rozstrzyga UP. Ten sam kierunek i zlecenie $5 wyceniono w T−59, bez aktualizacji p. Wiersze są wyłącznie kontrfaktyczną diagnostyką, nie transakcjami zamrożonej polityki. Opłat nie odejmowano drugi raz.

## Lejek kandydatów

| Model | Kandydaci | Dni | T−60 wycenialne | T−60 bez strony | Brak snapshotu T−60 | Brak snapshotu T−59 | T−59 niewykonalne | Wykonania T−59 | Wykonania rozliczone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | 8,675 | 45 | 7,409 | 1,266 | 0 | 0 | 53 | 7,356 | 7,356 |
| M0 | 8,675 | 45 | 7,409 | 1,266 | 0 | 0 | 57 | 7,352 | 7,352 |
| previous_D | 8,675 | 45 | 7,409 | 1,266 | 0 | 0 | 55 | 7,354 | 7,354 |

Predykcyjne EV i rzeczywiste PnL w tabelach porównawczych liczone są na tym samym podzbiorze: wycenione i wykonane w T−59 oraz rozliczone przed cutoff. Brak wyceny, wykonania lub rozliczenia pozostaje osobnym statusem i nie jest zerem.

## M1 według zamrożonych przedziałów EV

| Przedział EV T−60 | Kandydaci | Dni | T−59 wycenialne | Rozliczone | Śr. EV T−60 | Śr. EV T−59 | Śr. zwrot netto | Suma EV T−60 | Suma PnL | Śr. p strony | Win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EV ≤ 0% | 5,141 | 45 | 5,109 | 5,109 | -3.492% | -3.470% | -4.512% | $-892.136 | $-1152.613 | +0.5045 | +49.87% |
| 0% < EV < 0.5% | 232 | 32 | 231 | 231 | +0.137% | +0.150% | -13.915% | $+1.583 | $-160.717 | +0.5192 | +45.02% |
| 0.5% ≤ EV < 1% | 195 | 34 | 193 | 193 | +0.824% | +0.889% | +6.246% | $+7.947 | $+60.275 | +0.5211 | +54.92% |
| 1% ≤ EV < 2% | 950 | 43 | 942 | 942 | +1.700% | +1.693% | -0.894% | $+80.057 | $-42.106 | +0.5266 | +51.38% |
| 2% ≤ EV < 4% | 425 | 43 | 422 | 422 | +2.587% | +2.568% | -8.116% | $+54.593 | $-171.251 | +0.5239 | +47.16% |
| EV ≥ 4% | 466 | 34 | 459 | 459 | +6.400% | +6.390% | +0.598% | $+146.870 | $+13.733 | +0.5502 | +52.29% |

Wykres: [ev_return_by_bin.svg](ev_return_by_bin.svg). Słupki/linie nie są wygładzane ani wymuszane monotonicznie; rzeczywiste zwroty pokazują 95% przedziały bootstrapu blokowego.

95% CI dla rzeczywistego zwrotu i luki `R59 − E60` w każdym przedziale M1 (2,000 poprawnych replikacji na statystykę):

| Przedział EV | CI zwrotu netto | CI luki R−E60 |
|---|---:|---:|
| EV ≤ 0% | [-8.909%, -0.338%] | [-5.329%, +3.271%] |
| 0% < EV < 0.5% | [-23.146%, +0.752%] | [-23.283%, +0.607%] |
| 0.5% ≤ EV < 1% | [-6.649%, +24.377%] | [-7.495%, +23.575%] |
| 1% ≤ EV < 2% | [-6.322%, +3.081%] | [-8.058%, +1.382%] |
| 2% ≤ EV < 4% | [-18.242%, +3.273%] | [-20.760%, +0.704%] |
| EV ≥ 4% | [-9.131%, +10.061%] | [-15.479%, +3.533%] |

## Ranking okazji M1

| Grupa | n rozliczonych | Nachylenie: EV T−60 | 95% CI | Nachylenie: EV T−59 | 95% CI |
|---|---:|---:|---:|---:|---:|
| positive_EV | 2,247 | +0.6355 | [-1.6975, +2.6184] | +0.4499 | [-2.0064, +2.5045] |
| EV_ge_1pct | 1,823 | +0.3694 | [-1.8032, +2.2424] | +0.0471 | [-2.3408, +1.9928] |
| May_positive_EV | 1,406 | +0.9617 | [-1.7401, +3.0722] | +0.7997 | [-1.8358, +2.9148] |
| June_positive_EV | 841 | +0.4911 | [-19.2865, +12.3159] | -0.4054 | [-18.7076, +9.8033] |
| May_EV_ge_1pct | 1,031 | +0.6415 | [-1.9910, +2.5530] | +0.3457 | [-2.2820, +2.3351] |
| June_EV_ge_1pct | 792 | +2.9756 | [-17.5064, +19.5206] | +0.6699 | [-19.3296, +13.4055] |

Nachylenie to OLS z wyrazem wolnym: zrealizowany zwrot netto / $5 względem EV / $5. Nachylenie po zmianie ceny używa E59/$5, ale nadal zachowuje p i stronę z T−60.

## Rozdzielenie wykonania i reszty prognozy

Wartości są średnimi na jedno rozliczone wykonanie, w USD na żądane $5; przedziały podane są jako procent stawki. `Wynik − EV T−60 = (EV T−59 − EV T−60) + (y − p) × net shares T−59`.

| Grupa M1 | n | Luka R−E60 | Składnik wykonania | Reszta prognozy | 95% CI składnika wykonania | 95% CI reszty |
|---|---:|---:|---:|---:|---:|---:|
| EV > 0% | 2,247 | $-0.263 | $-0.000 | $-0.263 | [-0.013, +0.011]% | [-9.988, -0.708]% |
| EV ≥ 1% | 1,823 | $-0.264 | $-0.000 | $-0.263 | [-0.025, +0.005]% | [-10.682, -0.340]% |
| Maj, EV > 0% | 1,406 | $-0.363 | $-0.000 | $-0.363 | [-0.018, +0.015]% | [-13.556, -1.174]% |
| Czerwiec, EV > 0% | 841 | $-0.095 | $+0.000 | $-0.095 | [-0.016, +0.022]% | [-8.040, +2.519]% |
| Maj, EV ≥ 1% | 1,031 | $-0.355 | $-0.000 | $-0.354 | [-0.028, +0.012]% | [-14.627, +0.322]% |
| Czerwiec, EV ≥ 1% | 792 | $-0.146 | $-0.001 | $-0.145 | [-0.034, +0.005]% | [-11.440, +1.856]% |

Test tożsamości numerycznej: 22,062 porównywalnych wykonań; maksymalny błąd bezwzględny $1.78e-15; tolerancja $1e-07. Przedział reszty nie jest sam w sobie testem kalibracji.

## Odpowiedzi

1. Dla dodatniego EV nachylenie M1 względem T−60 wynosi +0.6355 (95% CI [-1.6975, +2.6184]); po przeliczeniu ceną T−59 wynosi +0.4499 (95% CI [-2.0064, +2.5045]).
2. Miesięczne nachylenia T−60 dla dodatniego EV mają ten sam dodatni kierunek: maj +0.9617 [-1.7401, +3.0722], czerwiec +0.4911 [-19.2865, +12.3159]. Po przeliczeniu ceną T−59 maj pozostaje dodatni (+0.7997), a czerwiec zmienia kierunek (−0.4054); wszystkie te przedziały obejmują zero.
3. Dla M1, EV ≥ 1%, średnia luka to $-0.264 na wykonanie: $-0.000 wynika ze zmiany warunków wykonania, a $-0.263 z reszty wyniku przy p z T−60. Maj: $-0.000 i $-0.354; czerwiec: $-0.001 i $-0.145.
4. Za mało informacji do rozróżnienia dodatniego rankingu od braku użytecznego rankingu; przedział obejmuje zero, więc nie jest to dowód braku sygnału. Ewentualny test kalibracji wymaga osobnego, zamrożonego planu.

Niepewność liczono przez wspólny dla modeli bootstrap kalendarzowy z 3-dniowych bloków UTC, obejmujący wszystkie dni od 1 maja do 30 czerwca, także bez kandydatów w grupie. Cel to 2,000 poprawnych replikacji na statystykę; wynik zapisuje ich faktyczną liczbę. Przedziały są warunkowe na wcześniej wybranych modelach i nie usuwają selekcji konfiguracji.

Manifest ma 61 dni kalendarzowych; wykonano 3,070 wspólnych losowań bloków przy seedzie 20261017. Szczegółowe CSV: `ev_bin_summary.csv`, `funnel_summary.csv`, `decomposition_summary.csv`, `ranking_slope_summary.csv` w tym folderze.

Diagnostyka nie uruchamia automatycznie kolejnego eksperymentu.
