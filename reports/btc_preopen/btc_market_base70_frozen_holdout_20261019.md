# Zamrożony transfer M0/M1 na holdout RAW BASE70

## Zakres i dopasowanie

To historyczny test przenośności. Holdout był wcześniej oglądany, a hipoteza powstała po jego analizie; wynik nie jest niezależnym potwierdzeniem przewagi.

Holdout obejmuje starty rynków `2026-07-01T00:00:00Z`–`2026-10-06T23:55:00Z` UTC, z wyłącznym cutoffem rozliczeń `2026-10-07T00:00:00Z`. Decyzja pierwszego rynku przypada na `2026-06-30T23:59:00+00:00`.
M0/M1 dopasowano dokładnie raz na `10,790` wspólnych, kwalifikujących się wierszach development OOF (`2026-04-15T17:05:00+00:00`–`2026-06-30T23:50:00+00:00`). Etykiety były oficjalne i zwolnione ściśle przed granicą fit; najpóźniejszy release to `2026-06-30T23:56:22+00:00`. SHA-256 próby kalibracyjnej: `2e017d413760a8a996fe7d7698d4022fd8bba28987aaa3007e0797b05d9ada06`; hash zapisanych wierszy fitu: `163900fa4be73e8957934dd308b95608df0d17f8009518b837caffd4c9df77d5`.

| Model | a | b | c | Cel fitu |
|---|---:|---:|---:|---:|
| M0 | -0.0246777478641 | 0.912229933049 | — | 0.692827683963 |
| M1 | -0.0250158848572 | 1.02670955494 | 0.452142019064 | 0.692460206879 |

Definicje pozostały bez zmian: `m=logit(p_market)`, `d=logit(p_RAW)-m`, `M0=sigmoid(a+b*m)`, `M1=sigmoid(a+b*m+c*d)`. Zachowano przycinanie `1e-6`, nieważony log loss, karę `0.001*((b-1)^2+c^2)` (w M0 bez `c`) i L-BFGS-B bez ograniczeń znaków. Współczynniki nie zmieniały się w holdoucie.

Poprzedni eksperyment development dopasowywał M0/M1 codziennie przed pierwszą decyzją dnia. Tutaj jeden fit na pełnych kwalifikujących się OOF przeniesiono bez aktualizacji przez cały okres holdoutu; to odrębny test od codziennego dopasowania development.

## Dostępność i wspólny kalendarz

| Etap | Rynki | Udział z 28 224 RAW |
|---|---:|---:|
| Zapisane predykcje RAW BASE70 | 28,224 | 100.0% |
| Prawidłowy `p_market` z walidowanego T−60 | 24,737 | 87.6% |
| Wspólny RAW + `p_market` i identyfikator rynku | 24,737 | 87.6% |
| T−59 snapshot na wspólnym kalendarzu | 24,737 | 87.6% |
| Oficjalne rozliczenie do cutoffu na wspólnym kalendarzu | 24,735 | 87.6% |
| Wspólne etykiety jakościowe | 24,735 | 87.6% |

Wspólna próbka obejmuje `24,737` decyzji od `2026-07-01 00:00:00+00:00` do `2026-10-06 23:55:00+00:00`. Brakujące `p_market` pozostawiono brakujące; nie wstawiano 0,5. `MARKET`, `RAW`, M0 i M1 używają identycznych rynków dla jakości i identycznego kalendarza decyzji dla ekonomii. Dostępność prognozy, wykonania T−59 i oficjalnego rozliczenia są raportowane oddzielnie.

## Jakość na oficjalnych etykietach

| Wariant | Wiersze | Log loss | Brier | AUC |
|---|---:|---:|---:|---:|
| MARKET | 24,735 | 0.691582 | 0.249234 | 0.520138 |
| RAW | 24,735 | 0.692480 | 0.249666 | 0.522495 |
| M0 | 24,735 | 0.691616 | 0.249248 | 0.520139 |
| M1 | 24,735 | 0.691540 | 0.249202 | 0.528420 |

Kontrast to pierwszy wariant minus drugi. Dla log loss/Brier wartości ujemne sprzyjają pierwszemu wariantowi; dla AUC dodatnie.

| Kontrast | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---:|---:|---:|
| M1_MINUS_M0 | -0.000076 [-0.000395, 0.000301] | -0.000047 [-0.000201, 0.000130] | 0.008281 [0.004070, 0.012345] |
| M1_MINUS_RAW | -0.000940 [-0.001794, -0.000289] | -0.000464 [-0.000886, -0.000143] | 0.005925 [0.001580, 0.011180] |
| M0_MINUS_MARKET | 0.000034 [-0.000111, 0.000187] | 0.000014 [-0.000057, 0.000091] | 0.000001 [-0.000000, 0.000002] |

Przedziały pochodzą ze sparowanego bootstrapu kołowych bloków 3-dniowych UTC: 2,000 wspólnych replikacji, seed `20261019`.

### Jakość według miesiąca UTC

Październik jest niepełny (1–6 października).

| Miesiąc | Wariant | n | Log loss | Brier | AUC |
|---|---|---:|---:|---:|---:|
| 2026-07 | MARKET | 6,795 | 0.691210 | 0.249062 | 0.511626 |
| 2026-07 | RAW | 6,795 | 0.693591 | 0.250220 | 0.511543 |
| 2026-07 | M0 | 6,795 | 0.691233 | 0.249068 | 0.511629 |
| 2026-07 | M1 | 6,795 | 0.691765 | 0.249316 | 0.521428 |
| 2026-08 | MARKET | 7,576 | 0.691360 | 0.249126 | 0.517733 |
| 2026-08 | RAW | 7,576 | 0.691884 | 0.249369 | 0.529240 |
| 2026-08 | M0 | 7,576 | 0.691280 | 0.249085 | 0.517732 |
| 2026-08 | M1 | 7,576 | 0.690861 | 0.248867 | 0.533512 |
| 2026-09 | MARKET | 8,639 | 0.692108 | 0.249485 | 0.526559 |
| 2026-09 | RAW | 8,639 | 0.691987 | 0.249420 | 0.528260 |
| 2026-09 | M0 | 8,639 | 0.692124 | 0.249492 | 0.526559 |
| 2026-09 | M1 | 8,639 | 0.691767 | 0.249312 | 0.531077 |
| 2026-10 (niepełny) | MARKET | 1,725 | 0.691387 | 0.249124 | 0.532118 |
| 2026-10 (niepełny) | RAW | 1,725 | 0.693189 | 0.250021 | 0.512438 |
| 2026-10 (niepełny) | M0 | 1,725 | 0.692056 | 0.249457 | 0.532118 |
| 2026-10 (niepełny) | M1 | 1,725 | 0.692490 | 0.249673 | 0.524882 |

## Ekonomia na wspólnym kalendarzu

Każdy wariant sam wybiera stronę i transakcje. Zastosowano żądany notional zakupu $5, próg net EV 1% notionalu, decyzję T−60, wykonanie T−59, zapisane depth/snapshoty, historyczny tryb opłat i cutoff rozliczeń. No-trade ma PnL `$0`.

| Wariant | Decyzje | Fill / settled / nierozliczone | Notional settled | Debet settled | Cash fee settled / unresolved | Token fee eq. settled / unresolved | PnL netto | PnL/notional | PnL/debet | EV T−60 na tych samych settled | PnL−EV | Nierozl. notional / debet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MARKET | 24,737 | 0 / 0 / 0 | $0.00 | $0.00 | $0.00 / $0.00 | $0.00 / $0.00 | $0.00 | n/a | n/a | $0.00 | $0.00 | $0.00 / $0.00 |
| RAW | 24,737 | 3,738 / 3,737 / 1 | $18,685.00 | $19,362.36 | $677.36 / $0.19 | $0.00 / $0.00 | $-546.22 | -2.92% | -2.82% | $1,242.61 | $-1,788.83 | $5.00 / $5.19 |
| M0 | 24,737 | 4 / 4 / 0 | $20.00 | $21.03 | $1.03 / $0.00 | $0.00 / $0.00 | $-21.03 | -105.16% | -100.00% | $0.36 | $-21.39 | $0.00 / $0.00 |
| M1 | 24,737 | 630 / 630 / 0 | $3,150.00 | $3,279.80 | $129.80 / $0.00 | $0.00 / $0.00 | $-446.52 | -14.18% | -13.61% | $234.58 | $-681.11 | $0.00 / $0.00 |

Sparowany dzienny PnL M1−M0: średnio `$-4.34`/dzień, 95% CI `$-9.23` do `$-0.55`; różnica sumaryczna `$-425.49`.
Sparowany dzienny PnL M1−RAW: średnio `$1.02`/dzień, 95% CI `$-4.43` do `$6.77`; różnica sumaryczna `$99.69`.

Przedziały PnL używają tych samych 2 000 replikacji 3-dniowych UTC co jakość; dni bez transakcji mają dzienny PnL 0.

### PnL i przepływy według miesiąca UTC

Październik obejmuje tylko 1–6 października. Opłaty tokenowe to informacyjny ekwiwalent USD i nie są ponownie odejmowane od gotówki.

| Miesiąc | Wariant | Fill / settled / unresolved | Notional settled | Debet settled | Cash fee settled / unresolved | Token fee eq. settled / unresolved | PnL settled | PnL/notional | PnL/debet | EV settled | Unresolved debet |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07 | MARKET | 0 / 0 / 0 | $0.00 | $0.00 | $0.00 / $0.00 | $0.00 / $0.00 | $0.00 | n/a | n/a | $0.00 | $0.00 |
| 2026-08 | MARKET | 0 / 0 / 0 | $0.00 | $0.00 | $0.00 / $0.00 | $0.00 / $0.00 | $0.00 | n/a | n/a | $0.00 | $0.00 |
| 2026-09 | MARKET | 0 / 0 / 0 | $0.00 | $0.00 | $0.00 / $0.00 | $0.00 / $0.00 | $0.00 | n/a | n/a | $0.00 | $0.00 |
| 2026-10 (niepełny) | MARKET | 0 / 0 / 0 | $0.00 | $0.00 | $0.00 / $0.00 | $0.00 / $0.00 | $0.00 | n/a | n/a | $0.00 | $0.00 |
| 2026-07 | RAW | 1,244 / 1,244 / 0 | $6,220.00 | $6,445.10 | $225.10 / $0.00 | $0.00 / $0.00 | $-509.47 | -8.19% | -7.90% | $394.66 | $0.00 |
| 2026-08 | RAW | 1,117 / 1,117 / 0 | $5,585.00 | $5,791.64 | $206.64 / $0.00 | $0.00 / $0.00 | $-24.35 | -0.44% | -0.42% | $565.28 | $0.00 |
| 2026-09 | RAW | 1,053 / 1,053 / 0 | $5,265.00 | $5,453.76 | $188.76 / $0.00 | $0.00 / $0.00 | $137.74 | 2.62% | 2.53% | $224.57 | $0.00 |
| 2026-10 (niepełny) | RAW | 324 / 323 / 1 | $1,615.00 | $1,671.86 | $56.86 / $0.19 | $0.00 / $0.00 | $-150.14 | -9.30% | -8.98% | $58.10 | $5.19 |
| 2026-07 | M0 | 0 / 0 / 0 | $0.00 | $0.00 | $0.00 / $0.00 | $0.00 / $0.00 | $0.00 | n/a | n/a | $0.00 | $0.00 |
| 2026-08 | M0 | 3 / 3 / 0 | $15.00 | $15.78 | $0.78 / $0.00 | $0.00 / $0.00 | $-15.78 | -105.20% | -100.00% | $0.30 | $0.00 |
| 2026-09 | M0 | 1 / 1 / 0 | $5.00 | $5.25 | $0.25 / $0.00 | $0.00 / $0.00 | $-5.25 | -105.04% | -100.00% | $0.06 | $0.00 |
| 2026-10 (niepełny) | M0 | 0 / 0 / 0 | $0.00 | $0.00 | $0.00 / $0.00 | $0.00 / $0.00 | $0.00 | n/a | n/a | $0.00 | $0.00 |
| 2026-07 | M1 | 228 / 228 / 0 | $1,140.00 | $1,186.12 | $46.12 / $0.00 | $0.00 / $0.00 | $-309.56 | -27.15% | -26.10% | $68.29 | $0.00 |
| 2026-08 | M1 | 245 / 245 / 0 | $1,225.00 | $1,278.03 | $53.03 / $0.00 | $0.00 / $0.00 | $-120.07 | -9.80% | -9.39% | $136.03 | $0.00 |
| 2026-09 | M1 | 128 / 128 / 0 | $640.00 | $665.16 | $25.16 / $0.00 | $0.00 / $0.00 | $27.41 | 4.28% | 4.12% | $25.38 | $0.00 |
| 2026-10 (niepełny) | M1 | 29 / 29 / 0 | $145.00 | $150.49 | $5.49 / $0.00 | $0.00 / $0.00 | $-44.31 | -30.56% | -29.45% | $4.88 | $0.00 |

### RAW: uzgodnienie do istniejącego ledgeru i rynki wyłączone

Pełny wynik RAW BASE70 służy wyłącznie jako punkt uzgodnienia; nie porównuję ograniczonego wyniku M1 z pełnopróbkowym `−$546.22`.

| Próbka RAW | Fill / settled / unresolved | Rozliczony PnL | Debet nierozliczony |
|---|---:|---:|---:|
| Poprzedni pełny holdout | 3,738 / 3,737 / 1 | $-546.22 | $5.19 |
| Istniejący ledger RAW na wspólnych rynkach | 3,738 / 3,737 / 1 | $-546.22 | $5.19 |
| Nowy replay RAW na wspólnych rynkach | 3,738 / 3,737 / 1 | $-546.22 | $5.19 |
| Wyłączone przez brak wspólnej dostępności | 0 / 0 / 0 | $0.00 | $0.00 |

Uzgodnienie wspólnego RAW przeszło dokładnie dla side, notionalu, debetu, opłat, wypłat, PnL i statusu rozliczenia. `PnL pełny = PnL wspólny + PnL rynków wyłączonych`: `$-546.22 = $-546.22 + $0.00`.

### Koncentracja wyniku

| Wariant | PnL | Pięć najlepszych dni | Udział pięciu dni w PnL |
|---|---:|---:|---:|
| MARKET | $0.00 | $0.00 | n/a |
| RAW | $-546.22 | $290.36 | -53.16% |
| M0 | $-21.03 | $0.00 | -0.00% |
| M1 | $-446.52 | $145.92 | -32.68% |

### Niezależny audyt gotówki

Każdy settled fill sprawdzono ponownie arytmetyką `Decimal` niezależną od agregacji raportu: `payout − cash_debit = net_pnl`; dodatkowo `cash_debit = notional + cash_fee` oraz strata nie przekracza debetu. Zastosowano poprawkę audytu przepływów z `a5fd05e`. Ekwiwalent fee potrąconego w tokenach pozostaje wyłącznie informacyjny.

| Ledger | Max różnica PnL | Max różnica debetu | Naruszenia limitu straty |
|---|---:|---:|---:|
| MARKET | $0.00 | $0.00 | 0 |
| RAW | $0.00 | $0.00 | 0 |
| M0 | $0.00 | $0.00 | 0 |
| M1 | $0.00 | $0.00 | 0 |
| Poprzedni pełny RAW BASE70 | $0.00 | $0.00 | 0 |

## Wnioski

- Czy przyrostowa jakość M1 względem M0 przeniosła się poza development? AUC wzrosło o `0.008281` (95% CI `[0.004069634328906677, 0.012345317019535179]`), ale przedziały różnic log loss `[-0.00039548273885928187, 0.00030092001760231575]` i Brier `[-0.0002009422286209317, 0.00013022410935818255]` obejmują zero. Transfer jest widoczny dla rankingu AUC, lecz nie wykazano poprawy log loss/Brier.
- Czy M1 poprawił RAW na wspólnej próbce? Tak w trzech metrykach; M1−RAW Δ log loss `-0.000940`, Brier `-0.000464`, AUC `0.005925`; wszystkie sparowane 95% CI wykluczają zero.
- Czy M1 zarobił po kosztach, czy ograniczył ekspozycję? M1 stracił `$-446.52` na `630` fillach, RAW `$-546.22` na `3,738`. Mniejsza strata łączna M1 wynika z dużo mniejszej liczby transakcji; jego PnL/notional (`-14.18%`) był gorszy niż RAW (`-2.92%`). To ograniczenie ekspozycji, nie dowód dodatniego edge. No-trade: `$0`.
- Czy modelowe EV lepiej odpowiadało realizacji? Nie w tym teście: średni bezwzględny błąd `PnL−EV T−60` na własnych settled fillach M1 wyniósł `$5.98`, RAW `$5.25`. Oba warianty miały dodatnie prognozowane EV, lecz ujemny PnL; zestawy wykonań są różne.

Holdout był ujawniony przed powstaniem hipotezy; nie dopasowywano współczynników po wyniku, nie zmieniano progów/stron/okresów i nie uruchamiano wariantu dziennego.

## Odtwarzalność

Konfiguracja `configs/research/btc_market_base70_frozen_holdout_20261019.json`; zamrożony manifest `reports/btc_preopen/btc_market_base70_frozen_holdout_20261019_manifest.json`. Predykcje, współczynniki, ledgery, bootstrap i raport są checkpointowane w `data/analysis/polymarket/BTC/btc_market_base70_frozen_holdout_20261019`. Wznowienie: `python run_btc_market_base70_frozen_holdout_20261019.py`. Nie trenowano BASE70, nie pobierano danych, nie odtwarzano archiwum ani nie zmieniano ustawień live.
Ostatnie wznowienie wykorzystało ukończone checkpointy; jego czas wyniósł `5.044372899996233` s, a szczytowy working set procesu wznowienia `1.558` GiB. Dopasowanie, bootstrap i ledgery ukończono we wcześniejszym przebiegu.
