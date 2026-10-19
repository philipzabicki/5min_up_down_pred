błąd raportowania/definicji stawki — księgowanie ledgerów jest poprawne; $5 oznacza notional zakupu, a dodatkowe opłaty gotówkowe podnoszą wypływ.

# Przyrostowa informacja RAW BASE70 ponad prognozę rynku

## Zakres i metoda

Jedyny eksperyment używa development `2026-04-15T17:05:00+00:00`–`2026-07-01T00:00:00+00:00`. Nie trenowano modelu BTC i nie odtwarzano archiwum. Historyczny holdout nie uczestniczył w eksperymencie; jego zapisany ledger odczytano wyłącznie do kontroli księgowej. Hipoteza powstała po wcześniejszej analizie holdoutu, więc eksperyment go nie ocenia.

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

Każdy wariant wybierał stronę przy zleceniu o notionalu $5 i progu 1% net EV. EV na T-60 to prawdopodobieństwo pomnożone przez tokeny po opłacie minus pełny wypływ gotówki; próg dzieli EV przez żądany notional $5. Na T-59 wykonanie zachowuje wybraną stronę i notional, a rzeczywisty wypływ wynosi notional plus opłata gotówkowa. W reżimie V1 opłata kupna pomniejsza tokeny; w V2 jest pobierana w gotówce. PnL przeliczono niezależnie jako wypłata minus wypływ gotówki.

| Wariant | Decyzje | Fill’e / rozliczone | Notional rozliczony | Wypływ gotówki | Opłaty gotówkowe | Ekwiwalent opłat tokenowych¹ | PnL | PnL / notional | PnL / wypływ | EV na rozliczonych fill’ach | PnL−EV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MARKET | 9,698 | 0 / 0 | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | n/a | n/a | $0.00 | $0.00 |
| RAW | 9,698 | 961 / 961 | $4805.00 | $4973.37 | $168.37 | $5.61 | $319.78 | 6.655% | 6.430% | $235.06 | $84.72 |
| M0 | 9,698 | 4 / 4 | $20.00 | $20.54 | $0.54 | $0.18 | -$20.54 | -102.700% | -100.000% | $0.23 | -$20.77 |
| M1 | 9,698 | 105 / 105 | $525.00 | $542.68 | $17.68 | $3.58 | $49.23 | 9.377% | 9.071% | $22.30 | $26.93 |

RAW anchor reconciliation: wcześniejsze `20,735` decyzje przyniosły `$267.44`. Na obecnym wspólnym kalendarzu stary ledger RAW dał `$319.78`, a nowy replay RAW `$319.78`; fill’e wyłączone przez wspólny kalendarz miały łącznie `-$52.34` PnL. Ledger RAW zgadza się na wspólnych rynkach.

Sparowana różnica średniego dziennego PnL M1−M0: `$1.01` na dzień (95% CI `-$0.58` do `$2.86`). Różnica sumarycznego PnL to `$69.77` (95% CI `-$40.15` do `$197.28`); przedział obejmuje zero.

¹ Informacyjny ekwiwalent USD opłaty potrąconej w tokenach; nie jest dodatkowym wypływem gotówki i nie odejmuje się go ponownie od PnL. Opłaty gotówkowe są już zawarte w wypływie i PnL.

### Cztery rozliczenia M0

Daty i wartości fill’i pochodzą z ledgeru; tokeny przed/po opłacie oznaczają odpowiednio `gross_shares` i `net_shares`. Ceny to VWAP wykonania.

| condition_id | Start UTC | Strona | Wynik rynku / zakład | Stawka | EV T−60 | EV / stawka | Notional zakupu | Wypływ gotówki | VWAP | Tokeny przed → po opłacie | Opłata pobrana | Ekwiwalent USD | Wypłata | PnL |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `0x7508407b8c8cc770696b4a2b18339d7753bb82e72049210660bb4f097b976c55` | 2026-04-28T07:55:00+00:00 | UP | DOWN; zakład przegrana | $5 | $0.052589676498538296 | 1.0517935300% | $5 | $5 | 0.48999999999999999 | 10.204081632653061 → 9.8293886326530604 | 0.374693 tokena | $0.18359957 | $0 | -$5 |
| `0x72946552fd2393ec94d76879791b5e7d9dea4ca12bc23a05bc357936bc66f235` | 2026-04-28T13:00:00+00:00 | UP | DOWN; zakład przegrana | $5 | $0.058410401423483194 | 1.1682080285% | $5 | $5.18 | 0.5 | 10 → 10 | $0.18 gotówką | $0.18 | $0 | -$5.18 |
| `0x884afc14b88dae3cbbbb4c4908d01467e72eeeb36c9a10979107d88a89d95103` | 2026-04-28T18:40:00+00:00 | UP | DOWN; zakład przegrana | $5 | $0.058410401423483194 | 1.1682080285% | $5 | $5.18 | 0.5 | 10 → 10 | $0.18 gotówką | $0.18 | $0 | -$5.18 |
| `0x28359384d25ed626c75bfb7c216edbbdc8a2dd1a330b5fb8f6b6100a72af7623` | 2026-04-28T23:55:00+00:00 | UP | DOWN; zakład przegrana | $5 | $0.058410401423483194 | 1.1682080285% | $5 | $5.18 | 0.5 | 10 → 10 | $0.18 gotówką | $0.18 | $0 | -$5.18 |

Ręczne uzgodnienie M0 z ledgeru: wypłaty `$0` − wypływy gotówki `$20.54` = PnL `-$20.54`. Notional wyniósł `$20`; opłaty pobrane w gotówce `$0.54`; opłata pobrana w tokenach miała ekwiwalent `$0.18359957` i nie stanowi dodatkowego wypływu. Stąd `-$20.54 = $0.00 - ($20.00 + $0.54)`. Zgodnie z wartościami zapisanymi w ledgerze suma ekwiwalentów opłat to `$0.72359957`: `$0.54` gotówką i `$0.18359957` informacyjnie w tokenach.

### Niezależna kontrola zapisanych ledgerów

Każdy settled trade przeliczono w osobnej ścieżce audytu z arytmetyką Decimal: `payout_usd - cash_debit_usd`. To niezależne od wspólnej funkcji używanej w agregatach głównej i miesięcznej tabeli. Warunek straty to `net_pnl_usd >= -cash_debit_usd`; gotówkowy wydatek sprawdzono też jako `gross_usd + fee_cash_usd`. Opłata w tokenach jest pokazana osobno jako ekwiwalent USD.

| Ledger | Fill’e / settled / unresolved | Notional settled | Wypływ settled | Gotówka unresolved | Opłaty cash (wszystkie fill’e) | Ekwiwalent opłat tokenowych | PnL ledgeru / cashflow | Max Δ PnL, USD | Max Δ debit, USD | Min. `PnL + wypływ` | Naruszenia limitu straty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bieżący MARKET | 0 / 0 / 0 | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 / $0.00 | $0 | $0 | $0.00 | 0 |
| Bieżący RAW | 961 / 961 / 0 | $4805.00 | $4973.37 | $0.00 | $168.37 | $5.61 | $319.78 / $319.78 | $2e-15 | $5e-16 | $0.00 | 0 |
| Bieżący M0 | 4 / 4 / 0 | $20.00 | $20.54 | $0.00 | $0.54 | $0.18 | -$20.54 / -$20.54 | $0 | $0 | $0.00 | 0 |
| Bieżący M1 | 105 / 105 / 0 | $525.00 | $542.68 | $0.00 | $17.68 | $3.58 | $49.23 / $49.23 | $2e-15 | $5e-16 | $0.00 | 0 |
| RAW BASE70 development | 1,038 / 1,038 / 0 | $5190.00 | $5367.94 | $0.00 | $177.94 | $10.90 | $267.44 / $267.44 | $2e-15 | $5e-16 | $0.00 | 0 |
| RAW BASE70 historyczny holdout | 3,738 / 3,737 / 1 | $18685.00 | $19362.36 | $5.19 | $677.55 | $0.00 | -$546.22 / -$546.22 | $2e-15 | $5e-16 | $0.00 | 0 |

Kontrola obejmuje wszystkie settled zakupy w bieżącym eksperymencie, RAW BASE70 development i ostatnim zapisanym holdoucie. W holdoucie jeden fill pozostaje nierozliczony: jego `$5.1925` wypływu gotówki jest pokazane osobno, bez dopisywania nieznanej wypłaty do PnL settled. We wszystkich settled ledgerach ręczne PnL zgadza się z kolumną ledgeru, wypływ z notionalem i opłatą gotówkową, a liczba naruszeń maksymalnej straty wynosi zero.

## Kontrakt `$5` i raportowanie

A (`$5` jako maksymalny wydatek z opłatami) nie opisuje tego kodu. B jest właściwą definicją: `$5` to żądany/gross notional zakupu; opłaty gotówkowe mogą zwiększyć całkowity wydatek. Historyczny tryb poboru opłaty zależy od daty: przed zmianą V1 potrąca opłatę z tokenów, po zmianie V2 pobiera ją w gotówce. Zmienia się jednostka opłaty, nie znaczenie stawki `$5`.

- Decyzja T-60 i EV: obliczenie EV odejmuje pełny `cash_debit_usd` od `p × net_shares`; próg 1% to net EV podzielone przez żądany notional `$5`.
- Wykonanie T-59: zachowuje wybraną stronę i notional; księguje gotówkową opłatę w `cash_debit_usd`, a V1 share fee w `net_shares`.
- Obrót: `gross_usd` to notional, nie wszystkie wydatki gotówkowe. `PnL / obrót` należy czytać jako `PnL / notional`; tabela pokazuje równolegle zwrot względem całkowitego wypływu.
- Rozliczenie i audyt: PnL to `payout_usd - cash_debit_usd`. USD-ekwiwalentu opłaty tokenowej nie odejmuje się ponownie.

### Ekonomia według miesiąca

| Miesiąc | Wariant | Fill’e / settled | Notional | Wypływ gotówki | Opłaty cash | Ekwiwalent opłaty tokenowej | PnL | PnL / notional | PnL / wypływ | PnL−EV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-04 | MARKET | 0 / 0 | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | n/a | n/a | $0.00 |
| 2026-05 | MARKET | 0 / 0 | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | n/a | n/a | $0.00 |
| 2026-06 | MARKET | 0 / 0 | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | n/a | n/a | $0.00 |
| 2026-04 | RAW | 34 / 34 | $170.00 | $170.55 | $0.55 | $5.61 | $4.49 | 2.642% | 2.633% | $0.80 |
| 2026-05 | RAW | 399 / 399 | $1995.00 | $2069.27 | $74.27 | $0.00 | $103.59 | 5.192% | 5.006% | -$34.88 |
| 2026-06 | RAW | 528 / 528 | $2640.00 | $2733.54 | $93.54 | $0.00 | $211.71 | 8.019% | 7.745% | $118.80 |
| 2026-04 | M0 | 4 / 4 | $20.00 | $20.54 | $0.54 | $0.18 | -$20.54 | -102.700% | -100.000% | -$20.77 |
| 2026-05 | M0 | 0 / 0 | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | n/a | n/a | $0.00 |
| 2026-06 | M0 | 0 / 0 | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 | n/a | n/a | $0.00 |
| 2026-04 | M1 | 38 / 38 | $190.00 | $193.22 | $3.22 | $3.58 | $21.26 | 11.192% | 11.006% | $18.34 |
| 2026-05 | M1 | 49 / 49 | $245.00 | $255.81 | $10.81 | $0.00 | $23.97 | 9.785% | 9.372% | $8.46 |
| 2026-06 | M1 | 18 / 18 | $90.00 | $93.65 | $3.65 | $0.00 | $3.99 | 4.434% | 4.261% | $0.12 |

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
Czas oryginalnego runnera: `68.11217820001184` s; szczytowy working set procesu: `1.865` GiB. Raport i audyt przeliczono z zapisanych ledgerów; nie pobierano danych, nie trenowano BTC ani nie odtwarzano archiwum.
