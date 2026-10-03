# Kontynuacja oceny nowego modelu BTC — 2026-10-03

Ten raport kontynuuje [ocenę modelu BTC](btc_new_model_evaluation_20261003.md) po commicie `327324a3b2278b63b799e90994319156fb44ba54`. Nie wykonywano nowego strojenia BTC. Wszystkie wyniki historyczne poniżej są diagnostyczne i nie stanowią niezależnego testu.

## 1. Skąd wzięło się $100 → $2,11

Odtworzenie księgi 1-sekundowego portfela `new_btc_platt` uzgadnia się bez błędu księgowania:

| Pozycja księgi | Wartość |
| --- | ---: |
| Kapitał początkowy | $100.00 |
| Transakcje / obrót | 73 / $365.00 |
| Opłaty zapisane w ledgerze | $14.70 |
| Otrzymane udziały | 834.1406 |
| Wypłaty z rozstrzygnięć | $267.1137 |
| Wynik po kosztach / saldo końcowe | **−$97.8863 / $2.1137** |
| Saldo zablokowane / otwarte pozycje na końcu | $0 / 0 |
| Największe obsunięcie kosztowe | 97.90% |

Stawka była stała i wynosiła $5 z góry, wraz z opłatą potrącaną przed wyliczeniem udziałów. Dla każdej pozycji `shares = (5 − fee) / ask`, a wypłata to liczba udziałów po właściwej stronie. Gotówka była blokowana do oficjalnego rozliczenia; bez długu i dopłat. Tożsamość `początkowy kapitał + PnL = saldo końcowe` jest spełniona.

| Strona | Transakcje | Wygrane | Obrót | Opłaty | PnL bez opłaty¹ | PnL po opłacie |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UP | 43 | 15 | $215.00 | $8.61 | −$43.7990 | −$50.5331 |
| DOWN | 30 | 9 | $150.00 | $6.09 | −$43.0143 | −$47.3533 |
| **Razem** | **73** | **24** | **$365.00** | **$14.70** | **−$86.8133** | **−$97.8863** |

¹ Kontrfaktyczne udziały liczone bez opłaty, przy tych samych wejściach i stawce $5. Opłata obniżyła wypłaty zwycięskich pozycji o $11.0729; przegrane i tak wypłacały zero. Koszty pogorszyły wynik, ale sam wynik bez opłat nadal wynosiłby −$86.81.

Model deklarował średnio 50.72% szansy dla wybranej strony i średnie EV +$0.66 na obserwowanym asku (+$0.79 po wycenie przy kwotowaniu wykonania opóźnionym o 1 s). Wynik rzeczywisty to 24/73 wygranych (32.88%) i −$1.34 na pozycję. Średni znormalizowany midpoint rynku dla wybranej strony wynosił 42.69%; model był średnio o 8.03 p.p. wyżej.

Kalibracja na rzeczywistych wejściach, w pięciu równolicznych grupach według `p_success`:

| Zakres przewidywania | n | Średnie p | Faktyczna częstość wygranej |
| --- | ---: | ---: | ---: |
| 0.461–0.491 | 15 | 47.93% | 13.33% |
| 0.491–0.501 | 14 | 49.57% | 21.43% |
| 0.501–0.512 | 15 | 50.60% | 46.67% |
| 0.512–0.522 | 14 | 51.56% | 50.00% |
| 0.522–0.566 | 15 | 53.91% | 33.33% |

W każdej grupie wynik był gorszy od deklarowanego, a najwyższa grupa nie była najlepsza. Portfel tracił kapitał głównie przez słabą jakość wybranych wejść. Stawka nie była dostosowana do malejącego kapitału: $5 stanowiło początkowo 5% kapitału, potem mediana 8.41%, kwartyl 75% 11.13%, a maksimum 73.37%. Po spadku gotówki poniżej $5 odrzucono 737 dalszych sygnałów z powodu braku środków. To wzmacniało wpływ błędnych wejść; nie wyjaśnia go samo w sobie.

Kontrola jakości wejść przy jednakowym nominale $5 i bez limitu portfela 100 USD dała 810 hipotetycznych transakcji: obrót $4,050, opłaty $166.34, deklarowane EV +$777.00, faktyczny wynik −$383.72, 318/810 wygranych (39.26%) i średnio −$0.47 na transakcję. Kontrfaktyczny wynik bez opłat to −$227.45. To diagnostyka jakości stawek wejściowych, nie portfel ani wykonalny bilans. Osobno w tym zbiorze: UP 178/463 wygranych, średnie `p_success` 50.91%, PnL −$283.18; DOWN 140/347, 50.20%, PnL −$100.54.

Wyniki całej zapisanej ścieżki portfela przy tej samej stawce i buforze EV:

| Wariant | Opóźnienie | Transakcje | Obrót | Opłaty | PnL | Saldo końcowe | Maks. DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MARKET_ONLY | 0 s | 7 | $35 | $1.16 | +$16.88 | $116.88 | 9.76% |
| MARKET_ONLY + BTC | 0 s | 7 | $35 | $1.20 | +$16.88 | $116.88 | 7.54% |
| BTC Platt | 0 s | 113 | $565 | $22.56 | −$98.17 | $1.83 | 98.17% |
| MARKET_ONLY | 1 s | 4 | $20 | $0.63 | +$5.24 | $105.24 | 4.54% |
| MARKET_ONLY + BTC | 1 s | 5 | $25 | $0.87 | +$0.24 | $100.24 | 6.44% |
| BTC Platt | 1 s | 73 | $365 | $14.70 | −$97.89 | $2.11 | 97.90% |
| MARKET_ONLY | 2 s | 6 | $30 | $1.00 | −$4.20 | $95.80 | 9.45% |
| MARKET_ONLY + BTC | 2 s | 6 | $30 | $1.03 | −$4.20 | $95.80 | 5.62% |
| BTC Platt | 2 s | 74 | $370 | $15.17 | −$97.13 | $2.87 | 97.30% |

## 2. MARKET_ONLY wobec MARKET_ONLY + BTC

Na 9,407 wspólnych oknach, przy decyzji bez dodatkowego opóźnienia, grupy wyglądały tak:

| Grupa decyzji | Liczba | Wynik równym nominałem $5 |
| --- | ---: | ---: |
| Ta sama strona i transakcja | 6 | 4 wygrane; łączny PnL +$21.88 |
| Dodana przez BTC | 1 | przegrana; −$5 |
| Pominięta przez BTC | 1 | przegrana; −$5 |
| Zmiana strony | 0 | — |
| Żaden wariant nie wchodzi | 9,399 | — |

W rzeczywistej ścieżce portfela przy opóźnieniu 1 s były cztery wspólne transakcje po tej samej stronie i jedna dodatkowa transakcja MARKET_ONLY + BTC; dodatkowa pozycja przegrała $5. MARKET_ONLY zamknął się na +$5.24, a wariant z BTC na +$0.24. Nie było transakcji wyłącznie MARKET_ONLY w tej ścieżce. Przy innych sekwencjach pojedyncza pozycja może zmienić gotówkę dostępną dla kolejnych wejść, dlatego różnica końcowych sald nie jest miarą jakości pojedynczego sygnału.

Średnie różnice strat `wariant z BTC − MARKET_ONLY` według grup (ujemna wartość sprzyja BTC):

| Grupa | n | Δ log loss | Δ Brier |
| --- | ---: | ---: | ---: |
| Brak transakcji w obu wariantach | 9,399 | −0.000227 | −0.000118 |
| Wspólna ta sama strona | 6 | −0.014827 | −0.005088 |
| Dodana przez BTC | 1 | +0.022097 | +0.010831 |
| Pominięta przez BTC | 1 | −0.014202 | −0.006950 |
| Wszystkie okna | 9,407 | −0.000235 | −0.000121 |

Prawie cała średnia poprawa całego zbioru przypada na 9,399 okien bez wejścia. Osiem okien z decyzją to zbyt mało, aby uznać poprawę jakości wejść za potwierdzoną.

W raporcie bazowym były już sparowane 3-dniowe moving-block bootstrapy (2,000 replikacji). Dla `MARKET_ONLY + BTC − MARKET_ONLY`: log loss −0.000235, 95% CI [−0.000464, −0.000064]; Brier −0.000121, 95% CI [−0.000231, −0.000035]. Są to przedziały z retrospektywnego okresu, który wpłynął na rozwój i wybór modeli; nie dowodzą niezależnej przewagi.

## 3. Rozbieżność offline/live i naprawa

Odtworzono próbkę obejmującą 1 października 22:44 UTC. Wcześniej cecha `ChaikinOsc_fit_1440m_...` wynosiła −141.1624186564 w ścieżce pseudo-live i −141.2069520573 w zapisanym offline; różnica wynosiła 0.0445334009, a różnica prawdopodobieństwa 0.00004468.

Źródłowe OHLCV było identyczne (`max abs diff = 0`), `float64`, bez braków i luk. Nie wykonywano resamplingu: `1440m` w nazwie opisuje horyzont celu użyty przy doborze wskaźnika. Różnicę powodowało obcięcie wejścia wskaźnika do okna per cecha (dla wskazanego Chaikina 7,166 świec), które resetowało stan AD/T3/GMA względem obliczenia offline z długą historią. Na tych samych danych okno 21,936 świec odtwarzało wartość Chaikina 1440m z błędem około 5.3e−8 wobec offline; pełny historyczny przebieg offline obejmował 3,319,992 świec.

`LivePredictor._resolve_indicator_window_len` używa teraz wspólnego maksymalnego bufora 21,936 dla wszystkich cech. Nie zmieniono tolerancji. Dla wskazanego Chaikina 1440m w 2026-10-01 22:44 UTC poprzednie okno 7,166 świec dawało różnicę 0.0445334009. Pełne okno 21,936 odtwarza tę cechę jako −141.2069520041 wobec offline −141.2069520573 (różnica około 5.3e−8); prawdopodobieństwo po poprawce wynosi dokładnie 0.45602524882881407.

Pełny pseudo-live replay 1,441 świec i 288 decyzji z 1–2 października trwał 61.9 s; peak working set 1,136,574,464 bajtów (około 1.06 GiB). OHLCV zgadzało się dokładnie, bez braków i luk. We wszystkich 288 punktach prawdopodobieństwo było identyczne z zapisanym offline, bez rozbieżności sygnału modelu; w danych brakowało współczesnych kwotowań Polymarket, więc oba porównywane policy outputs to `none` z powodem `missing_policy_input`, a nie rzeczywista walidacja wejść rynkowych.

Nie osiągnięto zgodności wszystkich wartości cech z pełną wieloletnią macierzą offline: maksymalna różnica wyniosła 89.6318436 dla cechy Chaikin 1440m 2 października 04:54 UTC; w żadnym z 288 punktów nie zmieniła ona predykcji. Pozostaje więc ograniczenie inicjalizacji na skończonej historii dla rekurencyjnego AD/Chaikin poza odtworzonym punktem 22:44. Zmiana usuwa rozbieżność konkretnego przypadku przez użycie wspólnej historii zamiast krótszego okna per cecha, ale ten replay nie dowodzi pełnej parzystości cech na całej osi czasu. Test restartu sprawdza odtworzenie wyniku od tych samych zachowanych świec; test uzupełniania luki sprawdza wymuszoną ciągłość i kolejność.

## 4. Zamrożony shadow bez zleceń

Protokół i collector: [protokół/config](../configs/btc_shadow_protocol_20261003.json), [zamrożone wagi drugiego poziomu](../configs/btc_shadow_frozen_models_20261003.json), [collector](../run_btc_shadow.py). Manifest uruchomionej sesji: [manifest](btc_shadow_manifest_20261003.json). Warianty to skalibrowany MARKET_ONLY, MARKET_ONLY + BTC oraz Platt BTC. Każdy ma osobne $100 wirtualnego salda, stałą hipotetyczną stawkę $5, bufor EV $0.25 i tę samą publiczną książkę. Zamrożona hipotetyczna opłata wynosi 7% zgodnie z odczytanym harmonogramem Gamma na starcie; późniejszy harmonogram jest rejestrowany, ale nie zmienia testu.

Collector używa tylko publicznych GET do Binance, Gamma i CLOB. Przechowuje świece, wspólne kwotowanie, bid/ask, widoczną głębokość, czas serwera, czas odbioru i wiek kwotowania; zapisuje predykcje i decyzje do SQLite jako zdarzenia niezmieniane po rozstrzygnięciu. Zakładane kupno po najlepszym asku, opłata i wypłata są jawnie hipotetyczne. `actual_fill` i `actual_fee` pozostają `unknown`. Zlecenia nie są wysyłane.

Collector uruchomiono 2026-10-03T20:17:55.654795Z (PID 27724); weryfikacja 20:28:34Z: proces dzia?a, catch-up doszed? do ?wiecy 20:27Z, zapisano 6 decyzji na 2 rynkach; settlement?w jeszcze nie by?o. Hash zbiorczy zamro?onych wej??: `2a106e7200e6f11cd903c640b79605339cefa60db197b2e2b759d2e5c3836752`; sesja zawiera 111 hashy plik?w. `trading_enabled=false`, `order_submission_count=0`.

Pierwsza przysz?a decyzja dotyczy rynku `btc-updown-5m-1791058800` (20:20?20:25Z). Czas odbioru ?wiecy BTC: 20:20:14.039Z; inferencja BTC: 20:20:14.070?20:20:14.364Z; CLOB odebra? kwotowania UP o 20:20:14.953Z i DOWN o 20:20:15.019Z; decyzje zapisano o 20:20:15.031Z. UP bid/ask = 0.63/0.64 (ask size 106.57), DOWN = 0.36/0.37 (ask size 2008.81); wiek timestamp?w ksi??ki 0.041/0.038 s. Zamro?ony model op?aty 7% zgadza? si? z harmonogramem Gamma.

| Wariant | p(UP) | Decyzja |
| --- | ---: | --- |
| MARKET_ONLY | 0.648711 | brak transakcji, EV poni?ej bufora |
| MARKET_ONLY + BTC | 0.653077 | brak transakcji, EV poni?ej bufora |
| BTC Platt | 0.531552 | hipotetyczny DOWN za $5 po $0.37; 12.9176 udzia?u; hipotetyczna op?ata $0.22050 |

To za?o?ona symulacja na widocznym asku i top-level size, nie potwierdzenie wykonania. `actual_fill=unknown`, `actual_fee=unknown`, a oficjalny wynik pierwszego rynku pozostawa? `unknown` przy tej weryfikacji; dwa hipotetyczne wej?cia BTC Platt by?y jeszcze otwarte (wirtualnie $90 got?wki i $10 zablokowane), bez rzeczywistych wype?nie?. Czas ko?ca predykcji w rekordzie obejmuje cechy i model BTC; warianty logistyczne rynku liczone s? po odebraniu kwotowania, przed zapisanym `decision_at_utc`. Pe?ne kwotowania, czasy, g??boko??, decyzje oraz p??niejsze rozstrzygni?cia zapisuje lokalna SQLite: `data/analysis/polymarket/BTC/shadow_20261003/shadow.sqlite3` (plik wykluczony z git). Manifest z hashami i czasem formalnej oceny: [manifest](btc_shadow_manifest_20261003.json). Start lub wznowienie: `python run_btc_shadow.py`.
## 5. Reguła przyszłej oceny

Główne porównanie to `MARKET_ONLY + BTC − MARKET_ONLY` w log loss i Brier na wspólnych, oficjalnie rozstrzygniętych przyszłych oknach. Niepewność będzie liczona sparowanym moving-block bootstrapem bloków 3-dniowych, 2,000 replikacji, seed `20261003`, 95% CI. Wynik portfela obejmie osobne wirtualne ścieżki i sparowany wynik hipotetyczny przy równym nominale na wspólnych kwalifikujących się oknach.

Minimalny czas to 730 dni i co najmniej 100 hipotetycznych pozycji combo kwalifikujących się do widocznego asku. Ocena formalna następuje po 730 dniach; przy mniejszej liczbie pozycji polityka pozostaje bez zmian do limitu 1,095 dni, po czym wynik z mniej niż 100 pozycjami jest nierozstrzygnięty. Nie ma wczesnego zatrzymania po dobrym wyniku ani zmian modelu, kalibracji, polityki, stawki lub kosztu w trakcie.

Wcześniejsza ścieżka miała 5 hipotetycznych transakcji combo w około 32.7 dniach (około 0.153 dziennie), czyli około 655 dni na 100 pozycji. Dwuletnie minimum daje około 112 pozycji przy tej samej częstości. Z przybliżenia błędu standardowego historycznego 3-dniowego CI oczekiwano by około 13,900 okien (około 50 dni) dla 80% mocy wykrycia historycznej różnicy LL; jest to optymistyczna ekstrapolacja z okresu developerskiego. Liczba pozycji, nie liczba okien, wyznacza długość testu.

Wynik pozytywny wymaga jednocześnie: spełnionego czasu i liczby pozycji, górnych granic 95% CI dla obu różnic strat poniżej zera, zyskownej i lepszej od MARKET_ONLY zamrożonej ścieżki combo oraz dodatniej dolnej granicy 95% CI dla sparowanego równostawkowego hipotetycznego PnL na wspólnym oknie. Wynik negatywny to wiarygodne pogorszenie obu metryk prognozy albo ujemny górny kraniec 95% CI dla sparowanego hipotetycznego PnL. Pozostałe wyniki są nierozstrzygnięte. Nawet dodatni shadow nie określi rzeczywistych wypełnień ani opłat.
