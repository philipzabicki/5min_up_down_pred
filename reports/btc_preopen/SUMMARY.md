# BTC Pre-Open v1: audit, economics, candidate study

Raport końcowy z audytu uruchomienia `btc_preopen_v1`, odtworzenia punktu w czasie, treningu kandydata i replayu publicznych zdarzeń order booka. Nie wykonano rzeczywistych zleceń ani aktywacji live.

## 1. Czy model wytrenowano poprawnie?

**Werdykt: poprawny z ograniczeniami.** Nie znaleziono potwierdzonego wycieku targetu ani przyszłych danych w użytych cechach w zweryfikowanej ścieżce treningu. Potwierdzono błąd w dawnym teście replay: dla świecy `Opened=17:03` przyciął on historię dopiero do `17:10`. To błędne potwierdzenie punktu w czasie; sam zapisany model pozostał niezmieniony.

Nowy replay z surowego prefiksu sprawdził 5 punktów, w tym granicę godziny i dnia. Każdy odtworzył **112/112 cech** oraz raw/Platt probability z maksymalną różnicą `0`; target nie był dostępny przy predykcji. Zasada czasu to świeca otwarta `T−2 min`, zamknięta `T−1 min`, decyzja `T−1 min`, rynek od `T` przez 5 minut; etykieta `Close[Opened+6m] >= Open[Opened+2m]`, remis UP, dostępna `Opened+7m`, UTC.

- **A — 181 vs 66 iteracji:** `181` to early-stopping punkt pojedynczego foldu. Końcowy wybór skanuje iteracje na foldach i minimalizuje średni log loss + `0,5 × odchylenie`; trial 30 osiągnął `0.692521820039`, a finalny model zachował jego parametry i 66 iteracji. Rozbieżność nie wskazuje na zły checkpoint.
- **B — 0.691942 vs 0.692147:** foldy, obserwacje i wagi są porównywalne; walidacja używa nieważonego log loss na minutach decyzyjnych. Selekcja cech dała `0.6919421` średniego LL i `0.6923578` po karze `0,5 × std`. Tuning wybrał `0.6925218`. Etap tuningu nie włączył selektora jako bazowego kandydata ani nie miał bramki akceptacji względem niego. To potwierdzona luka selekcji modelu.
- **C — przyczynowość replayu:** pięć skorygowanych odtworzeń daje identyczne cechy i predykcje; istniejący test perturbacji przyszłego okna również pozostaje w zestawie testów. Nie stwierdzono wpływu świec przyszłych na wcześniejszą prognozę w tych sprawdzeniach.
- **D — źródło etykiet:** historyczne `paired_uncertainty` w `evaluation.json` porównywało model z baseline na proxy Binance, nie na oficjalnym wyniku Polymarket. Poniżej i w `model_comparison.csv` log loss, Brier, AUC oraz paired 3-day block intervals są przeliczone osobno: proxy `n={proxy_rows}` i oficjalne Polymarket `n={official_rows}`. Etykiety rozeszły się w `400/{official_rows}` wspólnych rynków.
- **E — wagi:** `decision_weight=0.23`, `auxiliary_row_weight=0.1925` dobrano celowo przez balanced accuracy. To wybór metodologiczny, nie błąd implementacji; nowy ograniczony search dobiera wagę przez nieważony log loss minut decyzyjnych.

Model oryginalny: SHA-256 `ec5b0d8c3140f82713b33867075b2b384aad93e341369b7572401c4b8f3f7c20`, 112 cech, 66 iteracji, `2,925,500` wierszy treningowych, fit do `2026-01-01T00:00:00+00:00` (ostatnia dostępna etykieta `2025-12-31T23:59:00+00:00`). Kalibracja kończy się przed pierwszą decyzją testową. Okres testowy był wcześniej analizowany w repozytorium, dlatego nie jest pristine holdoutem.

## 2. Opóźnienia live i scenariusze czasu wejścia

Surowe pliki `data/live/BTC/trade/*.csv` i `data/live/BTC/logs/*.log` nie są obecne w checkoutcie; poniższy rozkład historycznego runtime pochodzi z utrwalonej tabeli audytu w `docs/polymarket_btc_experiment.md` (run `20260620_052109`, model różny od obecnego). Są to `p50/p95/p99` i liczebności zapisane w tym raporcie, nie ponownie przeliczone próbki.

| Etap z historycznego live | N | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| price_event_from_minute_open_ms (closed-candle close / following minute open; WS source event) | 527 | 4.00 ms | 14.00 ms | 17.74 ms |
| volume_event_from_minute_open_ms (closed-candle close / following minute open; WS source event) | 527 | 191.00 ms | 865.40 ms | 1540.12 ms |
| price_received_from_minute_open_ms (closed-candle close / following minute open; host wall clock) | 527 | 123.12 ms | 133.18 ms | 140.13 ms |
| volume_received_from_minute_open_ms (closed-candle close / following minute open; host wall clock) | 527 | 318.94 ms | 996.59 ms | 1667.34 ms |
| both_required_inputs_ready_from_minute_open_ms (closed-candle close / following minute open; host wall clock) | 527 | 321.38 ms | 996.59 ms | 1667.34 ms |
| feature_preparation_ms (local perf_counter) | 527 | 0.43 ms | 0.66 ms | 1.01 ms |
| feature_vector_construction_ms (local perf_counter) | 527 | 31.02 ms | 45.66 ms | 49.53 ms |
| model_inference_ms (local perf_counter) | 527 | 1.36 ms | 1.75 ms | 2.51 ms |
| signal_ready_from_window_start_ms (target window start, equal to closed-candle close in this live cycle; host wall clock) | 527 | 357.02 ms | 1033.15 ms | 1702.06 ms |
| quote_snapshot_lookup_or_refetch_ms (local perf_counter) | 527 | 0.07 ms | 0.13 ms | 53.06 ms |
| policy_decision_computation_ms (local perf_counter) | 526 | 0.09 ms | 0.17 ms | 0.21 ms |
| decision_ready_from_window_start_ms (target window start, equal to closed-candle close in this live cycle; host wall clock) | 526 | 357.50 ms | 1033.73 ms | 1780.79 ms |
| submit_call_including_response_ms (local perf_counter; synchronous call and response, no separate send/ACK) | 111 | 403.15 ms | 631.26 ms | 1001.86 ms |
| execution_stage_including_lookup_policy_and_submission_ms (local perf_counter; no-trade rows make median non-comparable to submitted calls) | 527 | 0.21 ms | 448.60 ms | 818.19 ms |
| cycle_complete_from_window_start_ms (target window start, equal to closed-candle close in this live cycle; host wall clock) | 527 | 475.28 ms | 1270.76 ms | 1791.68 ms |

W starym live kod uzywa `live_minute_opened = candle_opened + 1 min`, wiec kolumny opisane jako opoznienie od otwarcia minuty sa zakotwiczone w granicy zamkniecia wlasnie przetwarzanej swiecy. W tej sesji `p50` close-to-signal wynosil 357 ms, a `p50` close-to-cycle-end 475 ms, co wspiera obserwacje typowego cyklu ponizej sekundy. Ogon przekraczal sekunde: sygnal `p95=1.033 s`, cykl `p95=1.271 s`, `p99=1.792 s`. Czasy scienne hosta nie maja zapisanej kalibracji offsetu zegara. Koniec cyklu obejmuje wszystkie 527 decyzji, nie tylko 111 prob zlecenia; osobnego rozkladu close-to-send dla prob brak.
Treat the historical p50 values as usable timing-test references: signal-ready p50 is 357.02ms and cycle-complete p50 is 475.28ms (N=527). The `prestart_c0_o1` case rounds the latter up to a one-second entry offset at T-59s; it is not a p95/p99 or a measurement of the current candidate.

Z osobnego pre-open collection-only z 2026-10-04: decyzja nominalna była `T−60s`; pierwszy odbiór świecy zapisano `T−58.338663s`, gotową predykcję `T−58.314267s`, a wszystkie wejścia wykonawcze (w tym quote) `T−57.447320s`. To **jedna** obserwacja; druga ma predykcję gotową `T−57.336717s` i komplet wejść `T−56.339268s`. Dla dwóch próbek `p50` gotowości predykcji to 2.175 s po nominalnej decyzji, `p50` wszystkich wejść 3.107 s; `p95/p99` z `N=2` nie opisują wiarygodnie ogona. Zlecenia były wyłączone, więc liczebność send/ACK/fill wynosi zero.

Nowy kandydat LightGBM ma szybki, osobno zmierzony ciepły inference na 2,000 jednorzędowych wektorach 112 cech: p50 `0.026 ms`, p95 `0.030 ms`, p99 `0.071 ms` razem z kalibracją. Pomiar nie obejmuje budowy cech. Pełny inkrementalny update tych 112 cech nie jest obecnie podłączony do pre-open collectora: jego bundle ma 29 cech, a ogólny live runtime ma 256 kolumn, z których 44 pokrywają się z kandydatem. Nie mierzono odbudowy historii jako inferencji live.

Brak znaczników czasu wysłania, osobnego ACK giełdy i fillu. `submit_call_including_response_ms` (111 prób; `p50=403 ms`, `p95=631 ms`, `p99=1.002 s`) obejmuje synchroniczne wywołanie klienta z odpowiedzią; nie jest czasem fillu. Historyczny raport podaje dodatni `filled_stake_usdc` w 89 z 111 odpowiedzi, lecz nie zapisuje czasu ani niezależnego zdarzenia wykonania. W pre-open próbie nie było wysyłania zleceń.

The timing-grounded main economic scenario uses the existing `prestart_c0_o1` snapshot: the historical cycle-complete p50 of 475.28ms is rounded upward to one second, so entry is T-59s after the nominal T-60s candle-close decision. The signal-ready p50 is 357.02ms (N=527). These are historical timing references, not measurements of the current candidate runtime.
In the two pre-open collection observations, all execution inputs were ready +2.553s and +3.661s after the decision (N=2). The historical local submit-call p99 was 1.002s (N=111). The slower observed input time plus this local-call budget is 4.663s; rounding up to the existing five-second snapshot leaves 0.337s of margin.
The T-55s row is a conservative local-host component scenario, not a measured joint p99: the input and submit-call measurements come from separate runs and bundles. Submit duration includes the client response, with no separate exchange ACK or fill time. The candidate incremental feature path still has no matching live updater.
The T-60s row is an ideal reference; T-59s is the historical p50-based main scenario rounded upward from 475ms; T-58s is a +2s sensitivity. These old-live values are not current-candidate end-to-end measurements.

| Scenario | Dokladny czas wejscia | Model | Saldo koncowe | PnL netto | Obrot | Fee | Drawdown | Transakcje | Brak danych | Bez przewagi |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Main, historical p50 ceiling (475ms to 1s) (prestart_c0_o1) | T-59s | candidate_platt | $418.16 | $318.16 | $5380.00 | $0.50 | 40.67% | 1076 | 7076 | 1255 |
| Main, historical p50 ceiling (475ms to 1s) (prestart_c0_o1) | T-59s | candidate_raw | $455.53 | $355.53 | $5295.00 | $0.50 | 31.42% | 1059 | 7076 | 1272 |
| Main, historical p50 ceiling (475ms to 1s) (prestart_c0_o1) | T-59s | original_v1_platt | $4.89 | $-95.11 | $1435.00 | $0.00 | 96.75% | 287 | 7076 | 1208 |
| Main, historical p50 ceiling (475ms to 1s) (prestart_c0_o1) | T-59s | original_v1_raw | $0.57 | $-99.43 | $1865.00 | $28.32 | 99.61% | 373 | 7076 | 1035 |
| Ideal reference (prestart_c0_o0) | T-60s | candidate_platt | $412.59 | $312.59 | $5415.00 | $0.50 | 38.84% | 1083 | 7052 | 1272 |
| Ideal reference (prestart_c0_o0) | T-60s | candidate_raw | $435.36 | $335.36 | $5325.00 | $0.50 | 31.08% | 1065 | 7052 | 1290 |
| Ideal reference (prestart_c0_o0) | T-60s | original_v1_platt | $4.89 | $-95.11 | $1525.00 | $0.00 | 96.92% | 305 | 7052 | 1228 |
| Ideal reference (prestart_c0_o0) | T-60s | original_v1_raw | $3.87 | $-96.13 | $1970.00 | $28.80 | 97.51% | 394 | 7052 | 1051 |
| +2s sensitivity (prestart_c0_o2) | T-58s | candidate_platt | $419.05 | $319.05 | $5320.00 | $0.50 | 39.57% | 1064 | 7133 | 1210 |
| +2s sensitivity (prestart_c0_o2) | T-58s | candidate_raw | $456.43 | $356.43 | $5225.00 | $0.50 | 32.58% | 1045 | 7133 | 1229 |
| +2s sensitivity (prestart_c0_o2) | T-58s | original_v1_platt | $0.54 | $-99.46 | $1380.00 | $0.00 | 99.59% | 276 | 7133 | 1171 |
| +2s sensitivity (prestart_c0_o2) | T-58s | original_v1_raw | $1.16 | $-98.84 | $1845.00 | $28.30 | 99.15% | 369 | 7133 | 993 |
| Conservative local-host component (prestart_c0_o5) | T-55s | candidate_platt | $405.23 | $305.23 | $4955.00 | $0.50 | 42.58% | 991 | 7308 | 1108 |
| Conservative local-host component (prestart_c0_o5) | T-55s | candidate_raw | $423.94 | $323.94 | $4870.00 | $0.50 | 35.60% | 974 | 7308 | 1125 |
| Conservative local-host component (prestart_c0_o5) | T-55s | original_v1_platt | $1.46 | $-98.54 | $1270.00 | $0.00 | 98.99% | 254 | 7308 | 1066 |
| Conservative local-host component (prestart_c0_o5) | T-55s | original_v1_raw | $4.05 | $-95.95 | $2110.00 | $24.91 | 97.66% | 422 | 7308 | 903 |

All rows use ask age <=30s, capital release 60s after resolution, initial cash $100 and gross $5 per position. T denotes market start. The main historical p50-based entry is T-59s; T-60s is ideal, T-58s is +2s sensitivity, and T-55s is a conservative local-host component scenario.

Długie opóźnienia są osobnym testem odporności, nie głównym wynikiem: `prestart_c15_o5` wchodzi `T−40s`, `prestart_c45_o5` wchodzi `T−10s`. Nie należy z wyniku T−10s wnioskować o strategii wejścia T−60s.

## 3. Wynik ekonomiczny

Według schematu PMXT `timestamp_received` oznacza czas ingestu przez eksportera, a `timestamp` jest czasem źródłowym Polymarket; porównanie tych kolumn daje rozkład opóźnienia feedu.
Pobieranie odnotowało 4 ponownych prób dla godzin z błędem technicznym; szczegóły i wynik końcowy są zapisane w `audit.json`. Błędy HTTP nie są traktowane jako brak danych.
Wykonano pełny, checkpointowany odczyt archiwum PMXT v2 dla 9 407 oficjalnych rynków. Każdy hourly Parquet filtrowano do docelowych condition ID i zdarzeń odebranych nie później niż `market_start + 5s`. BBO odbudowano z pełnego snapshotu i aktualizacji; zakup przechodzi po ask przez poziomy wystarczające na $5. Użyto `timestamp_received` jako czasu dostępności. Brak pełnego snapshotu, brak historycznego `fee_rate_bps`, niekompletny book, przeterminowany ask lub niewystarczająca głębokość powodują wykluczenie.

Do replayu ekonomicznego wymagane są natywne snapshoty tokenów UP i DOWN oraz zgodność ask z ostatnim raportowanym BBO; komplementarne kwotowania służą wyłącznie do diagnostyki i nie dostarczają głębokości do fillu.
Przedziały wieku ask liczono po czasie źródłowym ostatniej zmiany głębokości ask; `timestamp_received` ogranicza dostępność, a jego wiek jest zapisany osobno w CSV.
PMXT nie zawiera monotonicznego identyfikatora kolejności zdarzeń; przy identycznym czasie odbioru i źródłowym zachowano kolejność w przefiltrowanym pliku Parquet, ale PMXT nie opisuje jej jako kolejności zdarzeń. Zgodność odtworzonego BBO jest raportowana, a niezgodność ask przy wejściu wyklucza snapshot.
Pilot wybrano chronologicznie dla pierwszego, środkowego i ostatniego rynku, niezależnie od predykcji i wyniku. Rzeczywiste odpowiedzi CLOB mapowały `condition_id` na tokeny UP/DOWN; próbki, pierwsze booki, eventy i ceny są w `audit.json`.

Zastosowano regułę fee zgodną z datą wejścia. Przed modernizacją giełdy 28 kwietnia 2026 r. opłata BUY była potrącana w tokenach wyniku (`shares × rate × min(price, 1−price) / price`); replay zaokrągla zagregowane poziomy booka w dół do 6 miejsc, bo PMXT nie udostępnia wypełnień per maker. Dla wejść od 11:00 do 12:00 UTC przyjęto okno konserwatywnej przerwy i nie symulowano zleceń. Od 12:00 UTC opłata jest w collateral: `shares × rate × (price × (1−price))`, wykładnik 1 i aktualna precyzja 5 miejsc/minimum $0.00001; historyczny wykładnik nie występuje w archiwum. Book agreguje rozmiary zamiast pokazywać wypełnienia makerów, więc cash fee zaokrąglono dla całego fillu, a legacy fee per poziom; dokładne zaokrąglenie każdego matchu jest nieznane. `fee_rate_bps` pochodzi z ostatniego odebranego eventu przed wejściem. To przybliżenie nie odtwarza dokładnej minuty wznowienia ani rozliczenia opłaty dla poszczególnych makerów. Źródła: [opis modernizacji](https://help.polymarket.com/en/articles/14762452-polymarket-exchange-upgrade-april-28-2026), [stary wzór kontraktu](https://github.com/Polymarket/ctf-exchange/blob/main/src/exchange/libraries/CalculatorHelper.sol), [nowy settlement](https://github.com/Polymarket/ctf-exchange-v2/blob/main/src/exchange/mixins/Trading.sol). Czas compute, order delay i dostępność środków po resolution są scenariuszami, nie pomiarami.

Coverage dla wariantow glownych (ask age <= 30s): `{"prestart_c0_o1": {"incomplete_or_crossed_book": 7000, "eligible": 2331, "stale_ask": 29, "unreconciled_best_ask_at_entry": 26, "exchange_maintenance_pause": 12, "no_initial_book_snapshot": 8, "unknown_historical_fee": 1}, "prestart_c0_o0": {"incomplete_or_crossed_book": 6977, "eligible": 2355, "stale_ask": 29, "unreconciled_best_ask_at_entry": 25, "exchange_maintenance_pause": 12, "no_initial_book_snapshot": 8, "unknown_historical_fee": 1}, "prestart_c0_o2": {"incomplete_or_crossed_book": 7036, "eligible": 2274, "unreconciled_best_ask_at_entry": 45, "stale_ask": 31, "exchange_maintenance_pause": 12, "no_initial_book_snapshot": 8, "unknown_historical_fee": 1}, "prestart_c0_o5": {"incomplete_or_crossed_book": 7233, "eligible": 2099, "stale_ask": 29, "unreconciled_best_ask_at_entry": 24, "exchange_maintenance_pause": 12, "no_initial_book_snapshot": 8, "unknown_historical_fee": 2}}`. Kazdy wariant startuje od $100, pojedynczy gross zakup to $5, a kapital wraca 60 s po `resolved_at_utc`. Wyniki dotycza tylko pokrytych rynkow; brakujacych bookow i nieznanych fee nie imputowano.

| Test odpornosci | Dokladny czas wejscia | Model | PnL netto | Obrot | Fee | Drawdown | Transakcje | Brak danych |
|---|---|---|---:|---:|---:|---:|---:|---:|
| prestart_c15_o5 | T-40s | candidate_platt | $277.36 | $4230.00 | $0.50 | 43.01% | 846 | 7612 |
| prestart_c15_o5 | T-40s | candidate_raw | $287.05 | $4165.00 | $0.50 | 34.25% | 833 | 7612 |
| prestart_c15_o5 | T-40s | original_v1_platt | $-99.20 | $3510.00 | $0.00 | 99.48% | 702 | 7612 |
| prestart_c15_o5 | T-40s | original_v1_raw | $-98.94 | $4095.00 | $21.47 | 99.43% | 819 | 7612 |
| prestart_c45_o5 | T-10s | candidate_platt | $82.09 | $1935.00 | $0.50 | 40.40% | 387 | 8569 |
| prestart_c45_o5 | T-10s | candidate_raw | $114.72 | $1890.00 | $0.50 | 36.19% | 378 | 8569 |
| prestart_c45_o5 | T-10s | original_v1_platt | $16.00 | $2110.00 | $0.00 | 67.01% | 422 | 8569 |
| prestart_c45_o5 | T-10s | original_v1_raw | $37.81 | $2410.00 | $10.29 | 49.76% | 482 | 8569 |

Fallback jest oddzielny: `market_start_c45_o0` wchodzi w T+0s (predykcja gotowa T-15s); `market_start_c45_o5` wchodzi T+5s. Pelna macierz opoznien, wieku ask i czasu zwolnienia kapitalu pozostaje w `economic_scenarios.csv`.

| Fallback | Dokladny czas wejscia | Model | PnL netto | Obrot | Fee | Drawdown | Transakcje | Brak danych |
|---|---|---|---:|---:|---:|---:|---:|---:|
| market_start_c45_o0 | T+0s | candidate_platt | $6.37 | $1295.00 | $0.00 | 57.62% | 259 | 8894 |
| market_start_c45_o0 | T+0s | candidate_raw | $38.67 | $1280.00 | $0.00 | 42.10% | 256 | 8894 |
| market_start_c45_o0 | T+0s | original_v1_platt | $8.61 | $1355.00 | $0.00 | 55.07% | 271 | 8894 |
| market_start_c45_o0 | T+0s | original_v1_raw | $9.97 | $1570.00 | $6.38 | 50.26% | 314 | 8894 |
| market_start_c45_o5 | T+5s | candidate_platt | $20.76 | $780.00 | $2.00 | 36.49% | 156 | 9134 |
| market_start_c45_o5 | T+5s | candidate_raw | $39.78 | $780.00 | $2.00 | 34.69% | 156 | 9134 |
| market_start_c45_o5 | T+5s | original_v1_platt | $10.50 | $765.00 | $2.00 | 38.81% | 153 | 9134 |
| market_start_c45_o5 | T+5s | original_v1_raw | $-8.62 | $850.00 | $4.00 | 47.96% | 170 | 9134 |

Coverage wariantu T+5s: `{"incomplete_or_crossed_book": 9067, "eligible": 273, "unreconciled_best_ask_at_entry": 26, "stale_ask": 20, "exchange_maintenance_pause": 12, "no_initial_book_snapshot": 8, "unknown_historical_fee": 1}`. Drawdown uwzglednia gotowke plus koszt pozycji zablokowanych, bez niezrealizowanej zmiany wartosci w trakcie rynku. Symulowany ask fill nie jest dowodem rzeczywistego wykonania.

Wcześniejszy `p_old_model_up` powstawał minutę później i miał dostęp do innej informacji, dlatego nie był porównywalny jako decyzja pre-open. Nie użyto polityki MARKET_ONLY.

Archiwum i schemat: [PMXT Polymarket v2](https://archive.pmxt.dev/Polymarket/v2), [PMXT v2 data overview](https://archive.pmxt.dev/docs/v2-data-overview). Fee formula i modernizacja: [Polymarket Trading Fees](https://help.polymarket.com/en/articles/13364478-trading-fees), [Exchange Upgrade April 28](https://help.polymarket.com/en/articles/14762452-polymarket-exchange-upgrade-april-28-2026), [legacy CalculatorHelper](https://github.com/Polymarket/ctf-exchange/blob/main/src/exchange/libraries/CalculatorHelper.sol), [v2 Trading settlement](https://github.com/Polymarket/ctf-exchange-v2/blob/main/src/exchange/mixins/Trading.sol). Raport zapisuje godziny, hash części, liczbę row groups, mapowanie tokenów i opóźnienie receive/source.

## 4. Wynik nowego treningu

Wytrenowano ograniczonego kandydata LightGBM GPU w 16 trialach, 4 wątkach i dwóch chronologicznych foldach. Kandydat i reguła wyboru zostały ustalone przed oceną Q3 2025; okres zewnętrzny nie służył wyborowi. Zwycięzca deweloperski: `tuned_search_winner`, historia 3 lata, waga decyzji `0.23`, 103 iteracje.

Na kwartale Q3 2025 log loss wyniósł `0.693225079` dla przepisu v1 i `0.693094880` dla kandydata. Sparowany 3-day bootstrap dla różnicy LL (kandydat − v1) ma 95% CI `[-0.0003506, 0.0000831]`, obejmujący zero. Na oficjalnych zewnętrznych etykietach przedział raw również obejmuje zero `[-0.0007251, 0.0000643]`. Proxy raw przedział to `[-0.0008618, -0.0001217]`.

Kandydat nie wykazał stabilnej poprawy według ustalonej reguły. Zachowano model v1 jako wybraną konfigurację; model kandydata pozostaje porównaniem badawczym. Raw/Platt oraz oficjalne/Proxy metryki i sparowane przedziały są rozdzielone w `model_comparison.csv`.

## Pliki dostawy

- `audit.json` — wynik audytu i dane źródłowe do odtworzenia werdyktu.
- `data_coverage.csv` — wszystkie 9 407 rynków dla 16 czasów wejścia.
- `model_comparison.csv` — metryki raw/Platt dla obu etykiet oraz paired intervals.
- `economic_scenarios.csv` i `economic_replay.json` — pełna macierz opóźnień, świeżości i zwalniania kapitału.
- `trades.parquet` — 109,551 wierszy dziennika transakcji dla scenariuszy z ask freshness do 30 s.
- `candidate_metrics.json`, `candidate_search_trials.csv` i pięć `point_in_time_*.json` — małe dowody treningu/replayu.
- `candidate_inference_latency.json`: warm single-row candidate inference and calibration; feature generation is not included.
- `report_bundle.zip` — raport, artefakty metadanych, wyniki i kod odtworzeniowy bez surowych shardów archiwum.

Nie aktywowano modelu live ani nie wysłano zleceń.
