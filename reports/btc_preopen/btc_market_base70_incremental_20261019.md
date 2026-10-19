# Przyrostowa informacja RAW BASE70 ponad prognozę rynku

## Zakres i metoda

Jedyny test używa development `2026-04-15T17:05:00+00:00`–`2026-07-01T00:00:00+00:00`. Nie wykorzystano holdoutu, nie trenowano modelu BTC i nie odtwarzano archiwum. Hipoteza powstała po wcześniejszej analizie holdoutu; ten eksperyment nie ocenia holdoutu.

Prawdopodobieństwo rynku to proporcja środków kwotowań UP/(UP+DOWN) na zwalidowanym T-60. Zmienna rynku to `m=logit(p_market)`, a różnica RAW to `d=logit(p_RAW)-m`; prawdopodobieństwa przycięto do `[1e-6, 1-1e-6]` przed logitem.

M0 ma postać `sigmoid(a+b*m)`. M1 ma postać `sigmoid(a+b*m+c*d)`. Osobne modele dopasowano raz dziennie na wszystkich wcześniejszych etykietach oficjalnych, których release był ściśle wcześniejszy niż pierwsza decyzja T-60 tego dnia. Oba modele korzystają z identycznych wierszy, co najmniej 1000 obserwacji i obu klas. Cel to średni, nieważony log loss plus `0.001*((b-1)^2+c^2)`; w M0 składnik c pominięto. Znaki współczynników nie były ograniczane.

BASE70 OOF i wcześniejsza selekcja cech widziały wszystkie etapy development. Dlatego dzienne dopasowanie jest przyczynowe względem etykiet, ale nie stanowi niezależnej oceny całego procesu budowy BASE70.

## Pokrycie i jakość prognoz

Wspólna próbka jakości: **9,694** rynków, `4,801` UP, `4,893` DOWN, od `2026-04-23T00:00:00+00:00` do `2026-06-30T23:50:00+00:00`. M0/M1 dopasowano w `69` z `77` dni UTC; rozgrzewka objęła `8` dni. Wspólny kalendarz ma `9,698` decyzji po rozgrzewce i dostępności obu prognoz.

| Wariant | Log loss | Brier | AUC |
|---|---:|---:|---:|
| MARKET | 0.692970 | 0.249909 | 0.513163 |
| RAW | 0.692114 | 0.249484 | 0.527526 |
| M0 | 0.693050 | 0.249948 | 0.518337 |
| M1 | 0.692558 | 0.249705 | 0.526950 |

| Kontrast (pierwszy minus drugi) | Δ log loss, 95% CI | Δ Brier, 95% CI | Δ AUC, 95% CI |
|---|---:|---:|---:|
| M1_MINUS_M0 | -0.000492 [-0.000778, -0.000209] | -0.000243 [-0.000384, -0.000106] | 0.008613 [0.002054, 0.015913] |
| M0_MINUS_MARKET | 0.000080 [-0.000273, 0.000414] | 0.000040 [-0.000136, 0.000206] | 0.005174 [-0.004276, 0.016308] |
| M1_MINUS_RAW | 0.000444 [-0.000076, 0.000976] | 0.000221 [-0.000039, 0.000488] | -0.000576 [-0.010749, 0.008447] |

Wartości ujemne dla log loss/Brier oznaczają przewagę pierwszego wariantu. Przedziały pochodzą ze wspólnego, sparowanego bootstrapu kalendarzowego z blokami 3 dni i 2000 replikacjami.

### Jakość według miesiąca

| Miesiąc | Wariant | Wiersze | Log loss | Brier | AUC |
|---|---|---:|---:|---:|---:|
| 2026-04 | MARKET | 1,023 | 0.692838 | 0.249846 | 0.487742 |
| 2026-04 | RAW | 1,023 | 0.691579 | 0.249216 | 0.539872 |
| 2026-04 | M0 | 1,023 | 0.692568 | 0.249710 | 0.478587 |
| 2026-04 | M1 | 1,023 | 0.691962 | 0.249408 | 0.524517 |
| 2026-05 | MARKET | 5,249 | 0.692817 | 0.249833 | 0.515309 |
| 2026-05 | RAW | 5,249 | 0.692309 | 0.249581 | 0.522702 |
| 2026-05 | M0 | 5,249 | 0.693094 | 0.249971 | 0.510905 |
| 2026-05 | M1 | 5,249 | 0.692723 | 0.249788 | 0.516825 |
| 2026-06 | MARKET | 3,422 | 0.693244 | 0.250043 | 0.513213 |
| 2026-06 | RAW | 3,422 | 0.691975 | 0.249415 | 0.539710 |
| 2026-06 | M0 | 3,422 | 0.693126 | 0.249985 | 0.505360 |
| 2026-06 | M1 | 3,422 | 0.692484 | 0.249667 | 0.529186 |

## Duże rozbieżności RAW–rynek

W górnym kwartylu początkowej różnicy logitów (próg `0.0818`, `2,424` wierszy) mediana odległości RAW od rynku wyniosła `0.1109`, a M1 `0.0372`. M1 był bliżej rynku w `97.7%` tych wierszy. To opisowa statystyka tej samej próbki development.

## Ekonomia przy stałym progu

Każdy wariant niezależnie wybierał stronę przy stałym zleceniu $5 i progu 1% net EV, z decyzją T-60, wykonaniem T-59 oraz istniejącym modelem opłat, głębokości i rozliczeń. PnL obejmuje rozliczone fill’e; fill’e nieukończone są raportowane osobno. Wszystkie warianty używają tego samego kalendarza decyzji.

| Wariant | Decyzje | Fill’e / rozliczone | Obrót rozliczony | Opłaty | PnL netto | PnL / obrót | EV na fill’ach rozliczonych | PnL−EV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MARKET | 9,698 | 0 / 0 | $0.00 | $0.00 | $0.00 | n/a | $0.00 | $0.00 |
| RAW | 9,698 | 961 / 961 | $4805.00 | $173.98 | $319.78 | 6.655% | $235.06 | $84.72 |
| M0 | 9,698 | 4 / 4 | $20.00 | $0.72 | -$20.54 | -102.700% | $0.23 | -$20.77 |
| M1 | 9,698 | 105 / 105 | $525.00 | $21.26 | $49.23 | 9.377% | $22.30 | $26.93 |

RAW anchor reconciliation: wcześniejsze `20,735` decyzje przyniosły `$267.44`. Na obecnym wspólnym kalendarzu stary ledger RAW dał `$319.78`, a nowy replay RAW `$319.78`; fill’e wyłączone przez wspólny kalendarz miały łącznie `-$52.34` PnL. Ledger RAW zgadza się na wspólnych rynkach.

Sparowana różnica średniego dziennego PnL M1−M0: `$1.01` na dzień (95% CI `-$0.58` do `$2.86`). Różnica sumarycznego PnL to `$69.77` (95% CI `-$40.15` do `$197.28`); przedział obejmuje zero.

| Wariant | Obrót z fill’i | Obrót rozliczony |
|---|---:|---:|
| MARKET | $0.00 | $0.00 |
| RAW | $4805.00 | $4805.00 |
| M0 | $20.00 | $20.00 |
| M1 | $525.00 | $525.00 |

### Ekonomia według miesiąca

| Miesiąc | Wariant | Fill’e / rozliczone | Opłaty | PnL netto | PnL / obrót rozliczony | PnL−EV |
|---|---|---:|---:|---:|---:|---:|
| 2026-04 | MARKET | 0 / 0 | $0.00 | $0.00 | n/a | $0.00 |
| 2026-05 | MARKET | 0 / 0 | $0.00 | $0.00 | n/a | $0.00 |
| 2026-06 | MARKET | 0 / 0 | $0.00 | $0.00 | n/a | $0.00 |
| 2026-04 | RAW | 34 / 34 | $6.16 | $4.49 | 2.642% | $0.80 |
| 2026-05 | RAW | 399 / 399 | $74.27 | $103.59 | 5.192% | -$34.88 |
| 2026-06 | RAW | 528 / 528 | $93.54 | $211.71 | 8.019% | $118.80 |
| 2026-04 | M0 | 4 / 4 | $0.72 | -$20.54 | -102.700% | -$20.77 |
| 2026-05 | M0 | 0 / 0 | $0.00 | $0.00 | n/a | $0.00 |
| 2026-06 | M0 | 0 / 0 | $0.00 | $0.00 | n/a | $0.00 |
| 2026-04 | M1 | 38 / 38 | $6.80 | $21.26 | 11.192% | $18.34 |
| 2026-05 | M1 | 49 / 49 | $10.81 | $23.97 | 9.785% | $8.46 |
| 2026-06 | M1 | 18 / 18 | $3.65 | $3.99 | 4.434% | $0.12 |

| Wariant | PnL 5 najlepszych dni | Udział w PnL ogółem |
|---|---:|---:|
| MARKET | $0.00 | n/a |
| RAW | $218.21 | 68.2% |
| M0 | $0.00 | -0.0% |
| M1 | $108.10 | 219.6% |

Wariant bez rozliczonych fill’i nie wykazuje przewagi po progu EV; próg pozostał bez zmian. Miesięczne zestawienie obejmuje również miesiące bez fill’i.

## Odpowiedź na pytania

- Czy RAW wnosi informację ponad skalibrowany rynek? Tak w jakości prognoz: M1 poprawia log loss o `0.000492` i Brier o `0.000243`, a oba 95% CI wykluczają zero. M1−RAW log loss wynosi `0.000444` (CI obejmuje zero), więc M1 nie wykazał przewagi nad samym RAW.
- Czy warstwa przewiduje przyczynowo późniejsze dni development? Tak, współczynniki dla każdego dnia używają wyłącznie etykiet zwolnionych przed pierwszą decyzją tego dnia; nie jest to niezależna ocena całego modelu BASE70.
- Czy korzyść utrzymuje się po kosztach i progu? M1 ma `$49.23` PnL wobec `-$20.54` dla M0; różnica M1−M0 to `$69.77`. Bootstrap CI sumarycznego PnL `-$40.15` do `$197.28` obejmuje zero, więc przewaga po kosztach nie jest rozstrzygająca.
- Czy M1 ogranicza duże rozbieżności? M1 był bliżej rynku w `97.7%` największych rozbieżności RAW–rynek.
- Jeśli M1 nie przewyższa M0, eksperyment kończy się tutaj; nie dodano drzew, interakcji ani siatki regularyzacji.

## Pliki i odtwarzalność

Manifest zamrożono przed obliczeniami: `reports/btc_preopen/btc_market_base70_incremental_20261019_manifest.json`. Dane prognoz, współczynniki dzienne, bootstrap, wyniki miesięczne i niezależne ledgery wariantów są w `data/analysis/polymarket/BTC/btc_market_base70_incremental_20261019`. Runner nie przyjmuje argumentów CLI: `python run_btc_market_base70_incremental_20261019.py`.
Czas runnera: `68.11217820001184` s; szczytowy working set procesu: `1.865` GiB. Nie pobierano danych, nie trenowano BTC ani nie używano holdoutu.
