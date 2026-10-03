# Ocena nowego modelu BTC — 2026-10-03

## Zakres i werdykt

Porównanie dotyczy 9,407 wspólnych, chronologicznych okien BTC 5m (2026-04-15T17:05:00+00:00 – 2026-05-18T10:30:00+00:00) na oficjalnym rozstrzygnięciu Polymarket. To wynik development, nie niezależny test: oceniane etykiety są w foldzie 8 nowego BTC i uczestniczyły w wyborze jego iteracji; fold walidacyjny selektora cech i strojenia profili również pokrywa ten okres. Nie trenowałem ponownie modelu BTC ani nie uruchamiałem szerokiego strojenia.

- Względem starego BTC: **lower log loss and Brier, but lower accuracy; retrospective development result** na wspólnych oficjalnych settlementach.
- Wartość ponad notowania rynku: **small lower log loss and Brier; development-only evidence** dla MARKET_ONLY + nowy BTC względem MARKET_ONLY.
- Wynik portfela po kosztach: **no robust incremental profitable value: at 1 s MARKET_ONLY ends at $105.24, MARKET_ONLY + new BTC at $100.24, and calibrated new BTC alone at $2.11** przy stałym $5, buforze EV $0.25 i odtworzonych kwotowaniach.
- Niezależny test: **nie istnieje w tych artefaktach**. Zalecany kolejny krok: shadow bez zleceń, z niezmienionym modelem i zapisem timestampów snapshotu/wykonania.

## Identyfikacja modelu i konfiguracji

Nowy model: `20261003_043549`, target `target_5m_candle_up`, 256 cech; stary model `20261002_041540`, 64 cech. Nowy meta ma pusty wektor `configured_constraints` i `applied_constraints`; plik modelu zapisuje pusty `monotone_constraints`. `monotone_constraints_method` i `monotone_penalty` w hiperparametrach nie aktywują ograniczeń bez wektora.

`feature_selection.exclusions_enabled=false`, `excluded_count=0`; artefakt zawiera 256 z 256 dostępnych cech. `modeling.json` zawiera listę wykluczeń, lecz flaga jest wyłączona. Walidacja konfiguracji wolumenu i reaction profile względem metadanych: True / True. Metadane modelu nie przechowują `config_snapshot` ani `config_path`; porównanie odtwarza zgodność efektywnych konfiguracji cech, lecz nie dowodzi hash całego pliku konfiguracji z chwili treningu.

Wspólne porównanie MARKET_ONLY odtwarza wcześniejszą definicję: standaryzowana regresja logistyczna L2 na 17 cechach jednoczesnej książki Kacho (bid/ask obu tokenów, midy, spready, sumy bid/ask, znormalizowany midpoint, logarytmy rozmiarów i imbalance rozmiarów). C wybierane jest ze zbioru `{0.01, 0.1, 1, 10}` na wcześniejszych danych. Wskaźnik `p_market_mid` jest tylko normalizowanym midpointem, nie ceną zakupu; transakcje używają ask.

Targety Binance i oficjalny settlement różnią się w 405 z 9,407 wspólnych okien (4.31%); dla szerszego zbioru nowego modelu: 691/15,677 (4.41%).

## 1. Jakość prognoz: stary BTC, nowy BTC i MARKET_ONLY

Główna część porównuje identyczne okna i ten sam moment obserwacji. Ostatnia kolumna pokazuje log loss w kolejnych outer foldach.

| Etykieta | Wariant | Okna | Log loss | Brier | AUC | Accuracy | Kalibracja: slope / intercept | Log loss F0 / F1 / F2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Oficjalny settlement Polymarket | Stary BTC (surowy OOF) | 9,407 | 0.692584 | 0.249717 | 0.5238 | 0.5165 | 0.621 / -0.002 | F0 0.693436 · F1 0.692416 · F2 0.691899 |
| Oficjalny settlement Polymarket | Nowy BTC (surowy OOF) | 9,407 | 0.691967 | 0.249411 | 0.5280 | 0.5143 | 0.741 / -0.003 | F0 0.693088 · F1 0.691754 · F2 0.691057 |
| Oficjalny settlement Polymarket | MARKET_ONLY (L2 logistic) | 9,407 | 0.688384 | 0.247547 | 0.5444 | 0.5273 | 0.923 / -0.012 | F0 0.688066 · F1 0.691973 · F2 0.685113 |
| Target BTC Binance (pomocniczo) | Stary BTC (surowy OOF) | 9,407 | 0.690812 | 0.248837 | 0.5368 | 0.5260 | 1.000 / -0.002 | F0 0.691640 · F1 0.690899 · F2 0.689896 |
| Target BTC Binance (pomocniczo) | Nowy BTC (surowy OOF) | 9,407 | 0.689868 | 0.248368 | 0.5436 | 0.5261 | 1.168 / -0.003 | F0 0.690950 · F1 0.689779 · F2 0.688874 |
| Target BTC Binance (pomocniczo) | MARKET_ONLY (L2 logistic) | 9,407 | 0.688112 | 0.247401 | 0.5482 | 0.5314 | 0.948 / -0.009 | F0 0.687450 · F1 0.690570 · F2 0.686315 |

Nowy BTC osobno na szerszym zbiorze OOF z oficjalnym settlementem: n=15,677, log loss=0.692282, Brier=0.249568, AUC=0.5255, accuracy=0.5137. To szersza retrospektywna ocena samego modelu; nie jest bezpośrednio porównywana z późniejszymi modelami drugiego poziomu.

Sparowany 3-dniowy moving-block bootstrap, 2,000 replikacji; jeden wspólny zestaw bloków dla wszystkich porównań:

- new_minus_old_log_loss: Δ log loss=-0.000617, 95% CI [-0.001089, -0.000143]
- new_minus_old_brier: Δ Brier=-0.000307, 95% CI [-0.000541, -0.000071]
- new_minus_market_only_log_loss: Δ log loss=+0.003583, 95% CI [+0.000619, +0.006382]
- new_minus_market_only_brier: Δ Brier=+0.001863, 95% CI [+0.000489, +0.003184]
- market_plus_new_minus_market_only_log_loss: Δ log loss=-0.000235, 95% CI [-0.000464, -0.000064]
- market_plus_new_minus_market_only_brier: Δ Brier=-0.000121, 95% CI [-0.000231, -0.000035]

## 2. Informacja ponad MARKET_ONLY

W MARKET_ONLY + nowy BTC dodano logit prawdopodobieństwa nowego modelu do tych samych 17 cech książki. Standaryzacja, C i dopasowanie regresji są wykonywane tylko na wcześniejszych oknach; outer okna porównania są wspólne.

| Wariant | Okna | Log loss | Brier | AUC | Kalibracja: slope / intercept |
| --- | ---: | ---: | ---: | ---: | ---: |
| Skalibrowany MARKET_ONLY | 9,407 | 0.688384 | 0.247547 | 0.5444 | 0.923 / -0.012 |
| Nowy BTC, Platt z wcześniejszych settlementów | 9,407 | 0.691907 | 0.249381 | 0.5281 | 1.295 / -0.018 |
| MARKET_ONLY + nowy BTC | 9,407 | 0.688148 | 0.247426 | 0.5475 | 0.946 / -0.011 |
| Dodanie nowego BTC vs MARKET_ONLY (różnica LL / Brier, 95% CI) | 9,407 | -0.000235 [-0.000464, -0.000064] | -0.000121 [-0.000231, -0.000035] | — | ujemna różnica sprzyja dodaniu |

## 3. Chronologiczne portfele startujące z $100

Stała stawka $5.00; zakup tylko gdy dokładny payoff po opłacie daje EV powyżej $0.25. Gotówka jest blokowana do `max(resolved_at_utc, market_end_utc)`, potem wpływ staje się dostępny; bez długu i dopłat. Ask z chwili decyzji zamraża stronę i limit. Przy opóźnieniu 1/2s wykonanie używa pierwszego zapisanego późniejszego asku nie wyższego od limitu; spadek ceny daje price improvement, brak dodatkowego slippage ponad limit. Książka Kacho ma nieznany wiek giełdowy. Opłaty pochodzą z zapisanych metadanych rynku; historycznych dat zmian opłat nie da się ustalić.

| Wariant | Opóźnienie | Końcowa gotówka | Kapitał otwarty | PnL | Maks. DD* | Obrót | Opłaty | Transakcje | Pominięte: brak gotówki |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| market_only | 0s | $116.88 | $0.00 | $16.88 | 9.76% | $35.00 | $1.16 | 7 | 0 |
| new_btc_platt | 0s | $1.83 | $0.00 | $-98.17 | 98.17% | $565.00 | $22.56 | 113 | 1187 |
| market_plus_new_btc | 0s | $116.88 | $0.00 | $16.88 | 7.54% | $35.00 | $1.20 | 7 | 0 |
| market_only | 1s | $105.24 | $0.00 | $5.24 | 4.54% | $20.00 | $0.63 | 4 | 0 |
| new_btc_platt | 1s | $2.11 | $0.00 | $-97.89 | 97.90% | $365.00 | $14.70 | 73 | 737 |
| market_plus_new_btc | 1s | $100.24 | $0.00 | $0.24 | 6.44% | $25.00 | $0.87 | 5 | 0 |
| market_only | 2s | $95.80 | $0.00 | $-4.20 | 9.45% | $30.00 | $1.00 | 6 | 0 |
| new_btc_platt | 2s | $2.87 | $0.00 | $-97.13 | 97.30% | $370.00 | $15.17 | 74 | 558 |
| market_plus_new_btc | 2s | $95.80 | $0.00 | $-4.20 | 5.62% | $30.00 | $1.03 | 6 | 0 |

*Obsunięcie jest liczone od gotówki plus kosztu otwartych pozycji, bez wyceny rynkowej. Na koniec wszystkie pozycje mają oficjalny settlement i koszt pozycji otwartych wynosi zero. Test nie modeluje nieznanej trwałości snapshotu ani rzeczywistego fillu.

## Jakość OOF, podziały i ograniczenia

Nowe OOF: 1,660,570 wierszy; zakres 2023-08-06 13:43:00+00:00–2026-10-02 17:52:00+00:00. Okres rynku 2026-04-15T17:05:00+00:00 – 2026-05-18T10:30:00+00:00 znajduje się w foldzie 8 nowego modelu.

- fold 0: train 2020-06-09 09:33:00+00:00–2023-08-06 13:42:00+00:00; test 2023-08-06 13:43:00+00:00–2023-11-29 21:19:00+00:00; label-unavailable rows at first prediction: 5; saved best iteration 2148.
- fold 1: train 2020-10-02 17:10:00+00:00–2023-11-29 21:19:00+00:00; test 2023-11-29 21:20:00+00:00–2024-03-24 04:56:00+00:00; label-unavailable rows at first prediction: 5; saved best iteration 1849.
- fold 2: train 2021-01-26 00:47:00+00:00–2024-03-24 04:56:00+00:00; test 2024-03-24 04:57:00+00:00–2024-07-17 12:33:00+00:00; label-unavailable rows at first prediction: 5; saved best iteration 1619.
- fold 3: train 2021-05-21 08:24:00+00:00–2024-07-17 12:33:00+00:00; test 2024-07-17 12:34:00+00:00–2024-11-09 20:10:00+00:00; label-unavailable rows at first prediction: 5; saved best iteration 2396.
- fold 4: train 2021-09-13 16:01:00+00:00–2024-11-09 20:10:00+00:00; test 2024-11-09 20:11:00+00:00–2025-03-05 03:47:00+00:00; label-unavailable rows at first prediction: 5; saved best iteration 1182.
- fold 5: train 2022-01-06 23:38:00+00:00–2025-03-05 03:47:00+00:00; test 2025-03-05 03:48:00+00:00–2025-06-28 11:24:00+00:00; label-unavailable rows at first prediction: 5; saved best iteration 1506.
- fold 6: train 2022-05-02 07:15:00+00:00–2025-06-28 11:24:00+00:00; test 2025-06-28 11:25:00+00:00–2025-10-21 19:01:00+00:00; label-unavailable rows at first prediction: 5; saved best iteration 1069.
- fold 7: train 2022-08-25 14:52:00+00:00–2025-10-21 19:01:00+00:00; test 2025-10-21 19:02:00+00:00–2026-02-14 02:38:00+00:00; label-unavailable rows at first prediction: 5; saved best iteration 1209.
- fold 8: train 2022-12-18 22:29:00+00:00–2026-02-14 02:38:00+00:00; test 2026-02-14 02:39:00+00:00–2026-06-09 10:15:00+00:00; label-unavailable rows at first prediction: 5; saved best iteration 1326.
- fold 9: train 2023-04-13 06:06:00+00:00–2026-06-09 10:15:00+00:00; test 2026-06-09 10:16:00+00:00–2026-10-02 17:52:00+00:00; label-unavailable rows at first prediction: 5; saved best iteration 976.

Dla wspólnych okien ostatnia etykieta treningowa foldu była dostępna 2026-02-14 02:44:00+00:00 przed pierwszą decyzją Polymarket z ocenianego zakresu. Na granicy każdego zapisanego foldu wykryto 5 wierszy, których target t+5m (dostępny w t+6m) nie był jeszcze dostępny przy pierwszej predykcji t+1m. Żadne z ocenianych okien rynkowych nie leży w tych pierwszych pięciu minutach foldu; same OOF-y z tych granic nie są użyte jako niezależne predykcje.

Najważniejsze ograniczenie: early stopping dla każdego zapisanego OOF foldu wybierał `best_iteration` na etykietach własnego testu (dla fold 8: 1326); końcowe 1528 drzew wybrano jako średnią najlepszych iteracji z 10 foldów. Fold 8 pokrywa oceniany okres, więc te OOF-y nie są niezależne od wyboru iteracji.

### Okresy strojenia i udział ocenianych danych

Zapisane parametry LGBM pochodzą z artefaktu `data/optuna/lgbm/BTC/lgbm_generic_optuna_best_mean_std_20260717_120022.json` (utworzony 2026-07-17T14:36:42.778515+00:00, trial 1/25); najlepsze parametry zgadzają się z metadanymi modelu. Zapis studium podaje 3,208,133 wierszy przed filtrem (aktywny: False), 10 foldów i liczbę wierszy, ale nie podaje zakresu timestampów ani hasha wejścia; zakres foldów poniżej jest rekonstrukcją na bieżącym zbiorze o zgodnej liczbie wierszy.
- LGBM Optuna (daty foldów odtworzone z bieżących danych), fold 9: walidacja 2026-03-26 20:47:00+00:00–2026-07-16 06:12:00+00:00 (train 2023-03-08 22:27:00+00:00–2026-03-26 20:46:00+00:00) obejmuje 47,126 wierszy ocenianego okresu 2026-04-15 17:04:00+00:00–2026-05-18 10:29:00+00:00.
- selekcja cech, fold 8: walidacja 2026-02-14 02:04:00+00:00–2026-06-09 09:34:00+00:00 (train 2022-12-18 22:14:00+00:00–2026-02-14 01:59:00+00:00) obejmuje 9,426 wierszy ocenianego okresu 2026-04-15 17:04:00+00:00–2026-05-18 10:29:00+00:00.
- volume profile Optuna, fold 8: walidacja 2026-02-14 02:04:00+00:00–2026-06-09 09:34:00+00:00 (train 2022-12-18 22:14:00+00:00–2026-02-14 01:59:00+00:00) obejmuje 9,426 wierszy ocenianego okresu 2026-04-15 17:04:00+00:00–2026-05-18 10:29:00+00:00.
- reaction profile Optuna, fold 8: walidacja 2026-02-14 02:04:00+00:00–2026-06-09 09:34:00+00:00 (train 2022-12-18 22:14:00+00:00–2026-02-14 01:59:00+00:00) obejmuje 9,426 wierszy ocenianego okresu 2026-04-15 17:04:00+00:00–2026-05-18 10:29:00+00:00.

Selekcja cech powstała 2026-10-03T01:59:12.272386+00:00: 886 wejściowych cech, 10 chronologicznych foldów, ranking recency-weighted/permutation; wybrano 256. Optuna volume profile (trial 205/500) i reaction profile (trial 98/500) miały po 10 foldów i używały targetu `target_5m_candle_up`; najlepsze konfiguracje zgadzają się z metadanymi nowego modelu.
Artefakty tych studiów nie zapisują hashy ani zakresów timestampów danych, dlatego okres foldów został odtworzony z aktualnego model-ready po zapisanych rozmiarach próbek i algorytmie walk-forward.

Dopasowanie wskaźników używa metryki `extremes_vs_mid_ir_oof` na 20 segmentach chronologicznych źródła 2020-06-09 09:33:00+00:00–2026-10-02 18:00:00+00:00; w każdym: 80% prefiksu, gap 2000 wierszy i metryka na końcowym fragmencie; wymagano 19 poprawnych segmentów.
- segment wskaźników 18: kalibracja progów 2026-02-14 02:46:00+00:00–2026-05-17 08:50:00+00:00 obejmowała 45,587 wierszy okresu oceny; gap obejmował 1,539 wierszy; walidacja metryki zaczynała się 2026-05-18 18:11:00+00:00 i obejmowała 0 wierszy oceny.


Liczebności: wejściowy Kacho miał 15,682 rynków; poprzedni pipeline zachował 15,677. Pozostałe 5 odrzucono: brak czasu rozstrzygnięcia: 2, pierwszy poprawny quote po limicie 2 s: 3. Dokładny join nowego OOF pokrył 15,677; brak połączeń: 0; połączenie nie używa pozycji wiersza. Próbka główna to 9,407 okien: pierwsze 6,270 służy jako wcześniejsze warm-up/tuning, a trzy outer foldy obejmują resztę. Kanoniczny OOF został nadpisany nowym treningiem; stare prawdopodobieństwa zachowano w zapisanym wyniku poprzedniego eksperymentu, którego manifest podaje oryginalny SHA-256 starego OOF.

Źródło decyzji to pierwszy poprawny snapshot Kacho po dostępności OOF; timestamp snapshotu jest równy lub późniejszy od decyzji, opóźnienie ma medianę 0 ms i maksimum 0 ms. Książka jest późniejszym zapisem historycznym z nieznanym wiekiem feedu. Prognozy i symulacja wykonania są więc oddzielnymi wynikami.

## Zgodność offline/live i następny krok

`run.py` ładuje model `20261003_043549` z `active.json`; ta konfiguracja istniała przed tym zadaniem i nie została przeze mnie zmieniona. Pseudo-live replay obejmował 2026-10-01T18:00:00–2026-10-02T17:59:00: 288 decyzji, 256 cech i rozgrzewkę 21936 świec (wymagane minimum 21936). OHLCV pochodził wyłącznie z lokalnego CSV; odchylenie od zapisanych OHLCV wyniosło 0.0. Kolejność cech zgadza się z metadanymi modelu (SHA-256 `7bf5ee7d9031d0a5415cdd5422c4992b561a7a3cead6f36c54a20782036fce35`); braki/statusy finite różniły się w 0 wartościach. Nie było rozbieżności sygnału (0) ani decyzji biznesowej (0).
Różnica predykcji max |Δp|=0.00004468, średnia 0.00000016; 1 wiersz(e) przekroczyły tolerancję 1.0e-06. Największy taki przypadek: 2026-10-01 22:44:00+00:00 (|Δp|=0.00004468, największa różnica cechy: ChaikinOsc_fit_1440m_pop128_fasmatypt3_fasper757_slomatypgma_sloper978_qe0.1_qm0.2_tf0.8_stmc_sg15). Replay wykorzystał zakotwiczone stany volume/reaction profile z lokalnej historii; nie używał REST.
Nie uruchamiałem aktywnego tradera, zleceń, redeemów ani wiadomości.

Werdykt: obecne dowody nie uzasadniają wniosku o dodatnim wyniku po kosztach ani zmiany podejścia modelowego na podstawie tego zbioru. Następny krok: non-trading shadow na przyszłych oknach, z zamrożonym modelem i polityką. W tym samym czasie zapisuj wiek/zmianę snapshotu, odbiór, wysłanie/ack/fill i rzeczywiste opłaty; po zebraniu nowych, nietkniętych etykiet wykonaj jeden niezależny test bez dostrajania progów.

Manifest: [`docs/btc_new_model_manifest_20261003.json`](btc_new_model_manifest_20261003.json). Artefakty wyliczenia: `data/analysis/polymarket/BTC/new_model_comparison/runs/d794ac2dea5a25a2`. Odtworzenie (bez argumentów CLI): `python run_btc_new_model_comparison.py`.
