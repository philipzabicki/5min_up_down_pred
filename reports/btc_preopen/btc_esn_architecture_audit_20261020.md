# Audyt ESN: przestrzeń architektur i selekcja portfela

## Odpowiedź

**Pierwotny eksperyment nie wykazał, że `ring 1×128` wygrał rzetelne porównanie rodzin i portfeli.** Silnik implementował single, parallel i DeepESN, a każda rodzina pojawiła się w 48 trialach. Przeszukiwanie nie gwarantowało jednak minimalnej liczby prób na rodzinę, a liczba jednostek w trialach nie była kontrolowana. Końcowy ranking skupił się na `single/ring/128`; zwycięski trial 44 nie należał do dwóch kandydatów sprawdzonych downstream w LightGBM. Portfel testował cztery dodatki z krótkiej listy i prawidłowo pozostał przy jednym członku według własnej reguły probe, ale to nie porównywało portfeli między architekturami.

Nowy, osobno wersjonowany search r4 porównuje trzy rodziny przy 128 lub 256 łącznych jednostkach, zaczyna od sześciu kontrolowanych konfiguracji z tymi samymi wejściami i sprawdza wkład kandydatów do pełnego zestawu. Wszystkie nowe wyniki w tym raporcie pochodzą wyłącznie z `search_validation` i nie stanowią niezależnego potwierdzenia poprawy.

Skorygowany etap wejść wskazał `single` z grupą `range_volatility` (probe Δ log-loss +0.000089797). Screen portfolio osiągnęło +0.000158274, lecz jego downstream LightGBM Δ log-loss wyniósł -0.000046807. Najlepszy pojedynczy downstream był `single/ring/128u` (returns,candle,range_volatility,volume,basis,time, Δ log-loss +0.000078732); wszystkie wyniki są selekcyjne dla tej samej `search_validation`.

## Zgodność pierwotnego eksperymentu

| Element planu | Zaimplementowany | Dostępny w searchu | Faktycznie oceniony | Dowód |
|---|---|---|---|---|
| Pojedynczy leaky ESN | Tak | Tak | Tak | `ESNConfig` i triale `single`; liczba w tabeli poniżej |
| Parallel ESN | Tak | Tak | Tak | Niezależne stany w `ESNEngine`; triale `parallel`; pełna liczba w eksporcie |
| DeepESN | Tak | Tak | Tak | Kolejna warstwa dostaje `next_states[r-1]`; triale `deep` |
| Ring i random sparse | Tak | Tak | Tak | Oba topologie w trialach; sparse ma losową maskę połączeń |
| Dobór wejść | Tak | Tak | Tak | Sześć binarnych grup wejściowych; zwycięzca użył grupy `volume` |
| Rozmiary reservoirów | Tak | 32, 64, 128 | Tak | Efektywne rozmiary w eksporcie; porównania nie wyrównywały sumy jednostek |
| Leak, recurrent/input scaling, bias | Tak | Tak | Tak | Ciągłe parametry Optuny; trzy wspólne seedy reservoiru |
| Connectivity | Tak | Dla sparse | Tak | Bernoulli dla połączeń poza diagonalą; diagonalę wymusza kod; ring ma 2 sąsiadów |
| Kompresja stanów | Tak | Stała | Tak | PCA 8 składowych na reservoir; 128 stanów ringa dawało 8 cech probe |
| Ocena na różnych seedach | Tak | 3 seedy reservoiru | Tak | Wspólne seedy `20261020–20261022`; osobne od seedów nie było w downstream |
| Selekcja portfela | Częściowo | Maks. 2 członków | Tak, ograniczona | Greedy probe testował 4 dodatki spośród 5 najlepszych; brak kosztu per dodatek i brak porównania portfeli rodzin downstream |

### Faktyczne pokrycie pierwszych 48 triali

| Rodzina/topologia | Próby | Ocenione probe | Odrzucone |
|---|---:|---:|---:|
| `single` | 31 | 31 | 0 |
| `parallel` | 10 | 10 | 0 |
| `deep` | 7 | 6 | 1 |
| `ring` | 25 | 25 | 0 |
| `sparse` | 8 | 8 | 0 |
| `dense` | 15 | 14 | 1 |

Wszystkie 48 rekordów Optuny mają stan `COMPLETE`; 47 miało wynik probe, a trial 1 został odrzucony: `reservoir saturation 0.281 exceeds frozen limit`. TPE nie wymuszało pokrycia rodzin. Pierwsze próby zawierały single, parallel i deep, lecz porównania np. 1×128 z 3×128 miały różne budżety. W próbach 19–44 dominował `single/ring/128`; 48 prób nie było z góry reprezentatywną pulą.

Pełne efektywne konfiguracje, parametry Optuny, statusy, wynik, czas, delty seedów, warm-up i przyczyny odrzucenia: [initial_48_trials.csv](btc_esn_architecture_audit_20261020/initial_48_trials.csv) oraz [initial_48_trials.json](btc_esn_architecture_audit_20261020/initial_48_trials.json). Eksport obejmuje 48/48 prób.

### Dokładna konstrukcja zwycięskiego `ring 1×128`

Trial 44: `architecture=single`, `topology=ring`, 128 jednostek, leak `0.694542963838798`, recurrent spectral radius `0.160627297354927`, input scaling `0.114581168795847`, bias scaling `0.285432156020883`. Wartość connectivity `0.15` jest nieużywana przez ring. Aktywacja to `tanh`; stan startuje od zera i jest resetowany po luce lub niepoprawnej świecy.

Dla jednostki `i` z indeksem modulo 128 aktualizacja ma postać:

```text
h_i(t) = (1 − leak) h_i(t−1) + leak · tanh(W_in,i · u(t) + b_i + W_rec,i,i−1 h_i−1(t−1) + W_rec,i,i+1 h_i+1(t−1))
```

Wiersz `i` przyjmuje sygnał z `i−1` i `i+1`; dwa wagi są losowane niezależnie z `Normal(0,1)`, po czym cała macierz jest skalowana przez `spectral_radius / max(abs(eigvals(W_rec)))`. Nie ma self-connections. Weights wejściowe i bias są losowane jednostajnie z `[-input_scaling,+input_scaling]` i `[-bias_scaling,+bias_scaling]`. Struktura sąsiedztwa ringa jest deterministyczna; wagi recurrent, input weights i bias zależą od seedu. PCA także zależy od seedu reservoiru.

Dwie surowe wejściowe cechy to `log_volume = log1p(Volume)` oraz `volume_ratio_to_prior_30m_median = Volume / median(poprzednich 30 wolumenów)`. Druga cecha wyklucza bieżącą świecę; luki resetują historię. Obie cechy są standaryzowane średnią i odchyleniem z fit-core; wartości niefinitywne są zastępowane zerem po transformacji.

Trzy seedy reservoiru: `20261020, 20261021, 20261022`. Screen delta log-loss (control minus augmented) dla trialu 44: `0.000048640, 0.000061983, 0.000055077`; średnia `0.000055233`, odchylenie populacyjne `0.000005448`. Warm-up (liczba minut po resecie) per seed: `63, 63, 63`; szczegóły diagnostyki są w JSON eksporcie.

128 surowych stanów było kompresowanych przez PCA fitowane na 20 000 stanów z fit-core; probe dostawał 8 składowych. Model downstream używał 70 kolumn BASE70 w kolejności z `B0_cv_summary.json`, a po nich 8 składowych `r0_pc0–r0_pc7` (nazwy `esn_32d4cf6804_seed_<reservoir_seed>_r0_pc0` … `_r0_pc7`). Czyli winner nie przekazywał 128 cech LightGBM.

BASE70 order: `candle_close_location_value_15m_lag1`, `rp_short_imbalance`, `candle_body_pressure_15m_lag1`, `rp_short_resistance_center_dist`, `candle_streak_15m`, `KeltnerChannel_fit_5m_preopen_v1_pop64_atrmatypfba_atrper1151_maper533_matypnwmaepan_souhlcc4`, `candle_streak_3m`, `candle_ret_co_5m_lag1`, `rp_short_support_center_dist`, `session_close_impulse_B3`, `realized_volatility_down_5m`, `candle_close_location_value_30m_lag1`, `candle_close_location_value_4h_lag1`, `candle_signed_vol_15m_lag1`, `candle_close_location_value_1h_lag1`, `candle_body_pressure_1h_lag1`, `realized_volatility_up_15m`, `futures_index_basis_rel_30m`, `session_open_impulse_NYSE`, `rp_medium_resistance_below`, `rp_all_resistance_below`, `realized_volatility_compression_expansion_1h_4h`, `futures_index_basis_rel_15m`, `candle_streak_5m`, `candle_body_pressure_30m_lag2`, `candle_body_pressure_4h_lag1`, `candle_close_location_value_4h_lag7`, `vp_short_log_density_ratio_to_current_bin_plus_2`, `candle_wick_asym_1h_lag1`, `candle_streak_30m`, `rp_medium_support_center_dist`, `candle_body_pressure_4h_lag7`, `ADX_fit_5m_preopen_v1_pop64_adxper452_atrper1990_matypadxkama_matypatrnwmatria_matypnegdmehma_matypposdmema_negdmper706_posdmper1716`, `candle_close_location_value_15m_lag2`, `session_close_impulse_BVC`, `candle_wick_asym_4h_lag6`, `futures_index_basis_rel_5m`, `realized_volatility_15m`, `candle_body_pressure_15m_lag2`, `candle_wick_asym_4h_lag5`, `realized_volatility_down_1h`, `candle_wick_asym_4h_lag1`, `futures_index_basis_rel_4h`, `rp_medium_resistance_center_dist`, `realized_volatility_up_1h`, `candle_wick_asym_1h_lag2`, `candle_close_location_value_30m_lag4`, `vp_short_local_above_below_volume_log_ratio`, `session_open_impulse_B3`, `candle_body_pressure_1h_lag2`, `vp_short_log_density_ratio_to_current_bin_minus_2`, `candle_wick_asym_30m_lag3`, `session_close_impulse_Xetra`, `realized_volatility_up_5m`, `rp_all_support_above`, `StochOsc_fit_5m_preopen_v1_pop64_faskper24_slodmatyptrima_slodper1052_slokmatypnwmabiw_slokper1892`, `BollingerBands_fit_5m_preopen_v1_pop64_maper328_masouhlcc4_matypshmma_stdmaper53_stdmatypkama_stdsouhlcc4`, `realized_volatility_down_15m`, `candle_wick_asym_1d_lag7`, `vp_short_log_density_ratio_to_current_bin_plus_1`, `candle_body_pressure_30m_lag5`, `rp_all_imbalance`, `candle_body_pressure_1d_lag2`, `candle_body_pressure_30m_lag1`, `taker_Z_1m`, `taker_V_1m`, `taker_Z_5m`, `taker_V_5m`, `taker_Z_15m`, `taker_V_15m`.

### Sposób wyboru i portfel w r3

Optuna maksymalizowała `logloss(control) − logloss(augmented)`: logistyczna korekta bazowego logitu kontra ta sama korekta rozszerzona o PCA ESN. To była poprawa ponad baseline, nie samodzielna jakość probe. Predykcje bazowe powstały z LightGBM fitowanego na fit-core, z seedem 20261015; korekta uczyła się na OOS correction-fit, a ocenę dawała na późniejszym `search_validation`. Zestaw feature’ów BASE70 i hiperparametry były wcześniej wybrane z pełnego development, więc wniosek jest warunkowy względem tego baseline’u.

Kontrola i augmented logistic używały tego samego `C=0.1`, solvera i wierszy. Finaliści probe byli oceniani na trzech wspólnych seedach reservoiru; nie były to trzy niezależne realizacje LightGBM. Downstream w LightGBM sprawdził triale 42 i 43, czyli najwyżej ocenione konfiguracje poza wybranym trialem 44; wybrany winner 44 nie miał bezpośredniego downstream check na `search_validation`. W starym downstream seed reservoiru i LightGBM był tym samym numerem, więc brakowało pełnego rozdzielenia tych źródeł wariancji.

| Zestaw przed | Dodany kandydat | Wynik przed | Wynik po | Przyrost | Koszt | Decyzja |
|---|---|---:|---:|---:|---|---|
| `[]` | trial 44 (`ring 1×128`) | 0.000000000 | 0.000055233 | 0.000055233 | nie zapisano | wybrano jako single-screen leader |
| `trial 44` | trial 42 `single/ring 1×128` | 0.000055233 | 0.000051312 | -0.000003921 | nie zapisano | odrzucono: dodanie obniżyło wynik pełnego zestawu |
| `trial 44` | trial 43 `single/ring 1×128` | 0.000055233 | 0.000050682 | -0.000004551 | nie zapisano | odrzucono: dodanie obniżyło wynik pełnego zestawu |
| `trial 44` | trial 45 `parallel/dense 2×32` | 0.000055233 | 0.000054705 | -0.000000528 | nie zapisano | odrzucono: dodanie obniżyło wynik pełnego zestawu |
| `trial 44` | trial 4 `single/dense 1×32` | 0.000055233 | 0.000027632 | -0.000027601 | nie zapisano | odrzucono: dodanie obniżyło wynik pełnego zestawu |

Reguła testowała cztery dodatki (trial 42, 43, 45 i 4); najlepszy zestaw dwuelementowy uzyskał `0.000054705`, poniżej wyniku single `0.000055233`. Portfel faktycznie sprawdzono w tej krótkiej liście i odrzucono dodatki na podstawie wyniku korekty, nie korelacji stanów. Brak zapisanego kosztu per dodatek ogranicza audyt koszt/korzyść.

| Downstream `search_validation` r3 | Średnia poprawa log-loss | Seedy LightGBM (w tej wersji zrównane z reservoir seeds) |
|---|---:|---|
| trial 42 | +0.000086480 | `+0.000195666, +0.000152881, -0.000089107` |
| trial 43 | +0.000077976 | `+0.000138984, +0.000146528, -0.000051584` |

## Poprawiony search r4 — wyłącznie `search_validation`

Podział zachowuje fit-core do `2024-07-01T00:00:00+00:00`, correction-fit do `2025-01-01T00:00:00+00:00` i walidację `2025-01-01T00:00:00+00:00–2025-07-01T00:00:00+00:00`. Cache zawiera 2,660,546 raw rows i 2,649,226 BASE70 rows, materializowane przed końcem walidacji; runner sprawdza każdą etykietę `< 2025-07-01T00:00:00+00:00`. Poza tym cutoff runner nie czyta danych ani etykiet.

Początkowe sześć prób porównuje single 1×128, parallel 2×64 i deep 2×64 dla ring oraz sparse, z tymi samymi 128 jednostkami, wszystkimi grupami wejść i wspólnymi parametrami dynamiki. Kolejne próby Optuny mają warunkowe `n_reservoirs`/`layers`, budżety 128/256 jednostek i dobierają dynamikę. Drugi etap, po zamrożeniu lidera każdej rodziny, osobno dobiera grupy wejściowe. Ring i sparse są równoważnymi wariantami topologii; sparse connectivity jest losowanym prawdopodobieństwem połączenia poza wymuszoną diagonalą.

| Architektura/topologia — wspólna pula | Trial | Δ log-loss probe | Rozrzut między 3 reservoir seeds |
|---|---:|---:|---:|
| `single/ring` | 6 | -0.000194617 | 0.000015549 |
| `single/sparse` | 7 | -0.000201626 | 0.000009130 |
| `parallel/ring` | 8 | -0.000186505 | 0.000011684 |
| `parallel/sparse` | 3 | -0.000185597 | 0.000017629 |
| `deep/ring` | 4 | -0.000149743 | 0.000037458 |
| `deep/sparse` | 5 | -0.000120757 | 0.000038285 |

### Najlepsi kandydaci według rodziny i topologii

| Rodzina/topologia | Trial | Jednostki | Wejścia | Probe Δ log-loss | SD między reservoir seeds |
|---|---:|---:|---|---:|---:|
| `deep/ring` | 48 | 128 | `returns, candle, range_volatility, volume, basis, time` | -0.000006147 | 0.000018383 |
| `deep/sparse` | 34 | 256 | `returns, candle, range_volatility, volume, basis, time` | -0.000026274 | 0.000048563 |
| `parallel/ring` | 53 | 128 | `returns, candle, range_volatility, volume, basis, time` | -0.000074351 | 0.000054298 |
| `parallel/sparse` | 9 | 256 | `returns, candle, range_volatility, volume, basis, time` | -0.000180747 | 0.000015789 |
| `single/ring` | 61 | 128 | `returns, candle, range_volatility, volume, basis, time` | -0.000167979 | 0.000056146 |
| `single/sparse` | 31 | 256 | `returns, candle, range_volatility, volume, basis, time` | -0.000111475 | 0.000017556 |

### Osobna optymalizacja grup wejściowych

| Rodzina | Trial | Wybrane grupy | Probe Δ log-loss | SD reservoir seeds |
|---|---:|---|---:|---:|
| `deep` | 4 | `basis` | +0.000078527 | 0.000011003 |
| `parallel` | 43 | `range_volatility, basis` | +0.000056151 | 0.000024624 |
| `single` | 14 | `range_volatility` | +0.000089797 | 0.000005543 |

Poprawione study ma 18 kotwic rodzina×grupa oraz 41 adaptacyjnych triali; spośród 59 ocenionych prób było 32 unikalnych efektywnych konfiguracji i 27 powtórzeń. Probe i downstream używają tych samych 8 składowych PCA na kandydata. Kontrola i augmented logistic mają jednakowe `C=0.1`, a wszyscy kandydaci portfela są oceniani na tym samym zestawie korekta/walidacja dla danego wspólnego reservoir seed.

### LightGBM downstream i portfel

Poniższe delty są obliczone wyłącznie na `search_validation`; Δ log-loss to baseline minus augmented, więc wartość dodatnia oznacza mniejszy log-loss augmented. Każdy finalist ma ten sam podzbiór wierszy, model baseline jest dopasowywany na fit-core, a reservoir seeds (`20261020–20261022`) są krzyżowane z odrębnymi seedami LightGBM (`20261101–20261103`). Wartości są wynikiem selekcji na tej samej walidacji, więc nie są niezależnym potwierdzeniem.

| Kandydat | Konfiguracja | Średnia Δ log-loss | SD wszystkich 9 par | Średnia Δ Brier | Δ AUC |
|---|---|---:|---:|---:|---:|
| `dcd078fc09b7478418912872` | parallel/sparse 256u, returns,candle,range_volatility,volume,basis,time | +0.000032020 | 0.000078448 | +0.000015811 | -0.0000625 |
| `b656b64aa9fe3df479819222` | deep/ring 128u, returns,candle,range_volatility,volume,basis,time | +0.000068828 | 0.000100812 | +0.000034163 | +0.0000535 |
| `9d1bec12441dc6a4946bc053` | deep/sparse 256u, returns,candle,range_volatility,volume,basis,time | +0.000041810 | 0.000082071 | +0.000020733 | -0.0000653 |
| `44e399a127401eb4e7ff0fa6` | single/ring 128u, returns,candle,range_volatility,volume,basis,time | +0.000078732 | 0.000039060 | +0.000039089 | +0.0001047 |
| `fba0a1f5ffa4de85911e4886` | single/sparse 256u, returns,candle,range_volatility,volume,basis,time | +0.000066810 | 0.000084117 | +0.000033006 | -0.0000560 |
| `4785383d825670bad214c285` | parallel/ring 128u, returns,candle,range_volatility,volume,basis,time | +0.000001196 | 0.000057389 | +0.000000467 | -0.0004399 |
| `80bc3eab86cb1ade78b095ae` | deep/ring 128u, basis | +0.000010110 | 0.000108197 | +0.000005089 | -0.0000679 |
| `61a793c855348e7728ef0ead` | parallel/ring 128u, range_volatility,basis | -0.000007840 | 0.000105441 | -0.000003854 | -0.0000971 |
| `06b7965395f85db9ec3b8b7f` | single/sparse 256u, range_volatility | +0.000011029 | 0.000062600 | +0.000005526 | +0.0000521 |
| `selected portfolio` | single/sparse 256u; deep/ring 128u | -0.000046807 | 0.000079616 | -0.000023256 | -0.0004834 |

| Bieżący zestaw | Dodany kandydat | Wynik przed | Wynik po | Przyrost | Koszt (sekundy) | Decyzja |
|---|---|---:|---:|---:|---:|---|
| `[]` | `parallel/sparse 256u (dcd078fc09b7478418912872)` | +0.000000000 | -0.000180747 | -0.000180747 | 0.173 | rejected_single_candidate_rule |
| `[]` | `deep/ring 128u (b656b64aa9fe3df479819222)` | +0.000000000 | -0.000006147 | -0.000006147 | 0.168 | rejected_single_candidate_rule |
| `[]` | `deep/sparse 256u (9d1bec12441dc6a4946bc053)` | +0.000000000 | -0.000026274 | -0.000026274 | 0.227 | rejected_single_candidate_rule |
| `[]` | `single/ring 128u (44e399a127401eb4e7ff0fa6)` | +0.000000000 | -0.000167979 | -0.000167979 | 0.145 | rejected_single_candidate_rule |
| `[]` | `single/sparse 256u (fba0a1f5ffa4de85911e4886)` | +0.000000000 | -0.000111475 | -0.000111475 | 0.157 | rejected_single_candidate_rule |
| `[]` | `parallel/ring 128u (4785383d825670bad214c285)` | +0.000000000 | -0.000074351 | -0.000074351 | 0.158 | rejected_single_candidate_rule |
| `[]` | `deep/ring 128u (80bc3eab86cb1ade78b095ae)` | +0.000000000 | +0.000078527 | +0.000078527 | 0.141 | not_selected_lower_score |
| `[]` | `parallel/ring 128u (61a793c855348e7728ef0ead)` | +0.000000000 | +0.000056151 | +0.000056151 | 0.183 | not_selected_lower_score |
| `[]` | `single/sparse 256u (06b7965395f85db9ec3b8b7f)` | +0.000000000 | +0.000089797 | +0.000089797 | 0.375 | selected_initial_candidate |
| `06b7965395f85db9ec3b8b7f` | `parallel/sparse 256u (dcd078fc09b7478418912872)` | +0.000089797 | -0.000105374 | -0.000195171 | 0.354 | rejected_nonpositive_marginal_or_seed_rule |
| `06b7965395f85db9ec3b8b7f` | `deep/ring 128u (b656b64aa9fe3df479819222)` | +0.000089797 | +0.000037025 | -0.000052771 | 0.277 | rejected_nonpositive_marginal_or_seed_rule |
| `06b7965395f85db9ec3b8b7f` | `deep/sparse 256u (9d1bec12441dc6a4946bc053)` | +0.000089797 | +0.000013877 | -0.000075920 | 0.355 | rejected_nonpositive_marginal_or_seed_rule |
| `06b7965395f85db9ec3b8b7f` | `single/ring 128u (44e399a127401eb4e7ff0fa6)` | +0.000089797 | -0.000098686 | -0.000188483 | 0.280 | rejected_nonpositive_marginal_or_seed_rule |
| `06b7965395f85db9ec3b8b7f` | `single/sparse 256u (fba0a1f5ffa4de85911e4886)` | +0.000089797 | -0.000060650 | -0.000150446 | 0.382 | rejected_nonpositive_marginal_or_seed_rule |
| `06b7965395f85db9ec3b8b7f` | `parallel/ring 128u (4785383d825670bad214c285)` | +0.000089797 | -0.000033225 | -0.000123022 | 0.309 | rejected_nonpositive_marginal_or_seed_rule |
| `06b7965395f85db9ec3b8b7f` | `deep/ring 128u (80bc3eab86cb1ade78b095ae)` | +0.000089797 | +0.000158274 | +0.000068477 | 0.402 | selected_best_positive_marginal |
| `06b7965395f85db9ec3b8b7f` | `parallel/ring 128u (61a793c855348e7728ef0ead)` | +0.000089797 | +0.000092961 | +0.000003165 | 0.460 | positive_but_not_best |
| `06b7965395f85db9ec3b8b7f,80bc3eab86cb1ade78b095ae` | `parallel/sparse 256u (dcd078fc09b7478418912872)` | +0.000158274 | -0.000063151 | -0.000221425 | 0.547 | rejected_nonpositive_marginal_or_seed_rule |
| `06b7965395f85db9ec3b8b7f,80bc3eab86cb1ade78b095ae` | `deep/ring 128u (b656b64aa9fe3df479819222)` | +0.000158274 | +0.000037543 | -0.000120731 | 0.527 | rejected_nonpositive_marginal_or_seed_rule |
| `06b7965395f85db9ec3b8b7f,80bc3eab86cb1ade78b095ae` | `deep/sparse 256u (9d1bec12441dc6a4946bc053)` | +0.000158274 | +0.000061633 | -0.000096641 | 0.842 | rejected_nonpositive_marginal_or_seed_rule |
| `06b7965395f85db9ec3b8b7f,80bc3eab86cb1ade78b095ae` | `single/ring 128u (44e399a127401eb4e7ff0fa6)` | +0.000158274 | -0.000024379 | -0.000182653 | 0.648 | rejected_nonpositive_marginal_or_seed_rule |
| `06b7965395f85db9ec3b8b7f,80bc3eab86cb1ade78b095ae` | `single/sparse 256u (fba0a1f5ffa4de85911e4886)` | +0.000158274 | -0.000006157 | -0.000164431 | 1.061 | rejected_nonpositive_marginal_or_seed_rule |
| `06b7965395f85db9ec3b8b7f,80bc3eab86cb1ade78b095ae` | `parallel/ring 128u (4785383d825670bad214c285)` | +0.000158274 | +0.000012597 | -0.000145677 | 0.877 | rejected_nonpositive_marginal_or_seed_rule |
| `06b7965395f85db9ec3b8b7f,80bc3eab86cb1ade78b095ae` | `parallel/ring 128u (61a793c855348e7728ef0ead)` | +0.000158274 | +0.000129135 | -0.000029139 | 0.620 | rejected_nonpositive_marginal_or_seed_rule |

Wybrany screen portfolio: 2 członków, score `+0.000158274`; wszystkie trzy probe seed delty były dodatnie. Downstream wybrany zestaw miał średni Δ log-loss `-0.000046807`, z ujemną średnią dla każdego reservoir seedu. Najlepszy pojedynczy downstream `single/ring/128u` z pełnymi wejściami osiągnął `+0.000078732` i dodatni wynik dla każdego reservoir seedu mimo probe `-0.000167979`. Rozbieżność pokazuje, że probe nie zachował rankingu LightGBM; wyniki nie są potwierdzeniem modelu.

## Koszt, wznowienie i artefakty

Nowy search ma twardy limit 14,400 s (4 h), z limitem do 5,400 s na architektury, 1,800 s na wejścia i rezerwą 5,400 s na finalistów/portfel. Łącznie faktycznie zapisano 10850.8 s.

| Etap | Sekundy / liczba |
|---|---:|
| `architecture_search` | 5439.800 |
| `downstream_lightgbm` | 1105.592 |
| `downstream_lightgbm_corrected` | 499.561 |
| `historical_trial_export` | 2.085 |
| `input_group_correction` | 1829.903 |
| `input_search` | 1887.876 |
| `portfolio_selection` | 1.697 |
| `portfolio_selection_corrected` | 10.185 |
| `report_and_manifest` | 0.017 |
| `search state generation / warm-up / PCA` | 6942.245 |
| `search logistic probes` | 22.331 |
| `search stage time outside trial durations` | 0.000 |
| `trial runtime beyond search timeouts` | 58.620 |
| `unique scored search configs / repeated trials` | 99 / 29 |
| `portfolio state generation` | 0.000 |
| `portfolio full-set probes` | 9.670 |
| `portfolio orchestration / I/O` | 0.509 |
| `finalist feature generation` | 224.302 |
| `LightGBM fits` | 269.663 |
| `downstream prediction / I/O / orchestration` | 5.354 |
| `valid-pool rejected / failed / resumed` | 6 / 3 / 5 |
| `invalidated input trials / scored / rejected` | 22 / 20 / 2 |
| `failed Optuna trial time` | 71.195 |
| `pretrial failed starts / seconds` | 1 / 3.376 |
| `historical export` | 2.085 |
| `report + manifest finalization` | 0.017 |

### Historia wznowień przed oceną poprawionej puli

1. `fixed anchor-count typing before any trial ran` — 0 ukończonych i 6 oczekujących kotwic; szczegóły: `brak błędu próby`.
2. `recorded failed pretrial start and report timing before scoring` — 0 ukończonych i 6 oczekujących kotwic; szczegóły: `brak błędu próby`.
3. `copy the ready mask before deleting Windows memmap work files` — 0 ukończonych i 5 oczekujących kotwic; szczegóły: `PermissionError WinError 32 on capture_ready.dat during temporary work cleanup`.
4. `retain the copied screen features after deleting memmap views` — 0 ukończonych i 5 oczekujących kotwic; szczegóły: `UnboundLocalError output after the cache write; no trial score persisted`.
5. `retry atomic JSON replacement after a transient Windows file lock` — 0 ukończonych i 5 oczekujących kotwic; szczegóły: `PermissionError WinError 5 replacing runtime_ledger.json during parallel-ring state generation`.

Pierwotny runtime ledger rozdzielał search `1721.5s`, probe/state generation łącznie w tej samej puli, portfel `111.9s`, downstream `115.3s`, finalne etapy `73.1s` oraz wcześniejsze nieudane próby `158.3s`. Stary runner nie rozdzielał kosztu generowania stanów i probe per trial ani kosztu każdego dodatku do portfela.

Polecenie uruchomienia i wznowienia (bez argumentów CLI): `python run_btc_esn_architecture_audit_20261020.py`. Nowy cache: `data/analysis/polymarket/BTC/btc_esn_architecture_audit_20261020_r4/`; manifest: `data/analysis/polymarket/BTC/btc_esn_architecture_audit_20261020_r4/audit_manifest.json`. Eksport pierwszych 48 triali: `reports/btc_preopen/btc_esn_architecture_audit_20261020/initial_48_trials.csv` i `reports/btc_preopen/btc_esn_architecture_audit_20261020/initial_48_trials.json`.

Holdout, późniejsze dane/etykiety/predykcje, wcześniejsza ocena po search-validation, live model i replay Polymarket nie są wejściem nowego runnera. Wyniki screeningu i finalistów na `search_validation` są selekcyjne.
<!-- corrected_input_and_boundary_diagnostics -->

## Korekta etapu wejściowego i presja granic hiperparametrów

Kontrola po zakończeniu pierwszego przebiegu wykazała niezgodność nazwy etapu: runner wywoływał `input_search`, a konfigurator sprawdzał wyłącznie `inputs`. Pierwsze 22 próby input-group nie zastosowały flag grup i użyły wspólnego pełnego zestawu wejść (20 ocenione, 2 odrzucone; koszt 1887.9 s). Zachowano je jako `input_search_search_uncorrected.json`, ale wyłączono z selekcji. Poniższe wyniki pochodzą z osobnego, poprawionego study.

Poprawione study: 59 prób, 59 ocenionych, 0 odrzuconych, 32 unikalnych efektywnych konfiguracji i 27 powtórzeń; kotwice grup/rodzin: 18/18; aktywny czas łącznie z wcześniejszym przebiegiem: 10850.8 s.

Zakres dla ciągłego parametru oznaczam jako wymagający sprawdzenia w szerszym zakresie, jeśli choć jeden z 10% najlepszych ocenionych triali znalazł się w zewnętrznych 5% przedziału. To sygnał presji samplera, nie dowód, że optimum leży poza zakresem. Connectivity liczę tylko na trialach sparse i ich własnym top decile.

| Parametr | Zakres | Zakres zaobserwowany | Blisko dolnej/górnej granicy: wszystkie | Top decile | Sygnał |
|---|---:|---:|---:|---:|---|
| `leak_rate` | [0.05, 1] | [0.0504103, 0.983201] | 24/1 z 69 | 7/0 z 7 | lower |
| `spectral_radius` | [0.1, 1.2] | [0.132572, 1.19649] | 1/12 z 69 | 0/1 z 7 | upper |
| `input_scaling` | [0.05, 1.5] | [0.0579033, 1.15147] | 4/0 z 69 | 0/0 z 7 | none |
| `bias_scaling` | [0, 0.5] | [0.00228504, 0.481984] | 1/1 z 69 | 0/0 z 7 | none |
| `connectivity` | [0.03, 0.5] | [0.0307291, 0.455053] | 6/0 z 31 | 2/0 z 4 | lower |

## Wskazówki dla następnego zakresu

Pierwotnie odnotowano presję dolnej granicy leak rate (7/7 najlepszych triali w dolnych 5%), sparse connectivity (2/4) oraz słabszy sygnał górnej granicy spectral radius (1/7). Szczegółowa kontrola rankingu poniżej zawęża tę interpretację: wszystkie 7 triali top decile to `deep/ring`, a wszystkie 4 najlepsze sparse triale to `deep/sparse`. Tych historycznych sygnałów nie traktuję jako wystarczającej podstawy do rozszerzenia zakresów.

Poprawiony etap można wznowić poleceniem `python run_btc_esn_input_group_correction_20261021.py`; pełny runner architektury: `python run_btc_esn_architecture_audit_20261020.py`.

## Zakres wyników

Wszystkie R4 probe i downstream wyniki pozostają selekcyjne dla `search_validation`. Nie są niezależnym potwierdzeniem. Holdout i późniejsze dane/etykiety nie zostały odczytane.

## Domknięcie interpretacji na podstawie kodu i zapisanych artefaktów — 2026-10-21

### Werdykt w skrócie

Najlepszy zapisany wynik downstream należy do kandydata `44e399a127401eb4e7ff0fa6`: study `btc_esn_architecture_audit_20261020_architecture`, trial 61, **single ESN / ring / 1×128 jednostek**, ze wszystkimi sześcioma grupami wejściowymi. Średnia sparowana downstream Δ log-loss wynosi `+0.000078731871` (raportowane `+0.000078732`).

Wybrany portfel probe składał się z `06b7965395f85db9ec3b8b7f` oraz `80bc3eab86cb1ade78b095ae`. Uzyskał `+0.000158274` w probe, ale `−0.000046807` downstream. Końcowy runner **nie przestawił wyboru na najlepszy model pojedynczy** po tej ocenie: zapisał wynik LightGBM portfela i kandydatów, lecz nie wykonał selekcji portfela według LightGBM.

### Zwycięzca downstream: dokładna konfiguracja

| Pole | Zapisana wartość |
|---|---|
| Kandydat / study / trial | `44e399a127401eb4e7ff0fa6` / `btc_esn_architecture_audit_20261020_architecture` / `61` |
| Rodzina / topologia | `single` / `ring` |
| Reservoiry / warstwy / jednostki | 1 reservoir, 1 warstwa, 128 jednostek |
| Grupy wejściowe | `returns`, `candle`, `range_volatility`, `volume`, `basis`, `time` |
| Parametry | leak `0.12875637661749686`; spectral radius `1.1513008725745195`; connectivity: nie dotyczy ring; input scaling `0.2580663119097347`; bias scaling `0.11590666009961913` |
| Seedy | reservoir `20261020, 20261021, 20261022`; LightGBM `20261101, 20261102, 20261103` |

21 unikalnych kolumn wejściowych pochodzących z tych grup:

- `returns`: `log_return_1m`, `log_return_5m`, `log_return_15m`, `log_return_60m`;
- `candle`: `candle_body_log_return`, `candle_close_location`, `candle_upper_wick_fraction`, `candle_lower_wick_fraction`;
- `range_volatility`: `candle_range_fraction`, `realized_volatility_5m`, `realized_volatility_15m`, `realized_volatility_60m`;
- `volume`: `log_volume`, `volume_ratio_to_prior_30m_median`;
- `basis`: `index_futures_log_basis`, `index_futures_basis_change_1m`, `index_futures_basis_change_5m`;
- `time`: `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`.

`FeatureScaler` wyznacza średnią i odchylenie standardowe z finite wartości wejściowych w `fit_core`, przed `2024-07-01T00:00:00Z`; zerowy scale zastępuje jedynką, a wartości niefinitywne po transformacji ustawia na zero. Warm-up mierzy zbieżność stanu z początkową perturbacją `0.25`, progiem względnej różnicy p95 `0.001` i stabilnym oknem 60 minut. Zapisane wartości dla trzech seedów reservoiru to **338, 163 i 130 minut**; wszystkie trzy diagnostyki przeszły próg.

Downstream nie dostaje 128 surowych stanów. Dla każdego seedu PCA z losowanym solverem jest fitowane na 20 000 stanów z `fit_core` i redukuje stan do 8 składowych. LightGBM dostaje `BASE70 + 8 PC = 78 cech`. Wagi fitu to `0.59` dla wierszy decyzyjnych i `0.1025` dla pozostałych minut. Parametry bazowego LightGBM pochodzą z `B0_cv_summary.json`: `gbdt`, `binary`, `num_leaves=81`, `learning_rate=0.06418546619439842`, `feature_fraction=0.7962744599393133`, `feature_pre_filter=false`, `min_data_in_leaf=12089`, `lambda_l1=9.508616956284627`, `lambda_l2=30.729843292781943`, `max_bin=63`, `device_type=gpu`, `gpu_use_dp=false`, `num_threads=8`, `metric=None`, `verbosity=-1`. Każdy wariant używał **31 boosting rounds bez early stopping**; seedy LightGBM nadpisywały pola `seed`, `data_random_seed`, `feature_fraction_seed` i `bagging_seed`.

Wybrany probe-owo portfel miał dwa człony: `06b7965395f85db9ec3b8b7f` — single/sparse, 1×256, grupa `range_volatility`, leak `0.1406152569426814`, radius `1.0052597180390357`, connectivity `0.4198237615858855`, input scaling `0.8775438223278621`, bias `0.36220759482074777`; oraz `80bc3eab86cb1ade78b095ae` — deep/ring, 2×64, grupa `basis`, leak `0.05687055713912644`, radius `0.9457661960099929`, input scaling `0.23391119636151508`, bias `0.37981034069596603`. Portfel łączy 384 jednostki reservoirów, 8 PCA cech na człon i `BASE70 + 16 = 86` kolumn w LightGBM. Connectivity nie dotyczy członu ring. Warm-up członów per reservoir seed wynosił odpowiednio `162/168/157` i `710/290/430` minut. Portfel używał tego samego crossed grid seedu reservoiru i LightGBM co zwycięzca single.

Target to `target_preopen_5m_up_binance_index_proxy`: `Close[Opened+6m] >= Open[Opened+2m]`, pięć świec 1m, remisy jako UP; decyzja nominalna `Opened+1m`, etykieta dostępna `Opened+7m`. Downstream używał fitu 2 123 739 wierszy oraz wspólnej maski 52 120 wierszy decyzyjnych z `search_validation`, od `2025-01-01` do `2025-07-01` exclusive. Wagi dotyczą fitu; raportowany validation log-loss jest nieważoną metryką tych samych wierszy. Δ log-loss oznacza `baseline − variant`, więc wartość dodatnia jest lepsza.

Poprawa `+0.000078732` to **średnia arytmetyczna dziewięciu różnic log-loss dla 3×3 sparowanych modeli**, a nie wynik pojedynczego seedu, średnia predykcji ani log-loss modelu ensemble. Baseline zależy od seedu LightGBM; ten sam baseline seed jest porównywany z każdym z trzech seedów reservoiru.

### Seed × seed: zwycięzca i portfel

| Wariant | Seed reservoiru | Seed LightGBM | Baseline LL | Wariant LL | Δ LL |
|---|---:|---:|---:|---:|---:|
| single/ring | 20261020 | 20261101 | 0.692359019 | 0.692294172 | +0.000064847 |
| single/ring | 20261020 | 20261102 | 0.692334181 | 0.692286249 | +0.000047931 |
| single/ring | 20261020 | 20261103 | 0.692493228 | 0.692371244 | +0.000121984 |
| single/ring | 20261021 | 20261101 | 0.692359019 | 0.692277529 | +0.000081490 |
| single/ring | 20261021 | 20261102 | 0.692334181 | 0.692309583 | +0.000024598 |
| single/ring | 20261021 | 20261103 | 0.692493228 | 0.692388328 | +0.000104900 |
| single/ring | 20261022 | 20261101 | 0.692359019 | 0.692338951 | +0.000020068 |
| single/ring | 20261022 | 20261102 | 0.692334181 | 0.692215648 | +0.000118533 |
| single/ring | 20261022 | 20261103 | 0.692493228 | 0.692368993 | +0.000124235 |
| portfel wybrany probe (downstream) | 20261020 | 20261101 | 0.692359019 | 0.692511474 | −0.000152455 |
| portfel wybrany probe (downstream) | 20261020 | 20261102 | 0.692334181 | 0.692473236 | −0.000139055 |
| portfel wybrany probe (downstream) | 20261020 | 20261103 | 0.692493228 | 0.692462902 | +0.000030326 |
| portfel wybrany probe (downstream) | 20261021 | 20261101 | 0.692359019 | 0.692391595 | −0.000032575 |
| portfel wybrany probe (downstream) | 20261021 | 20261102 | 0.692334181 | 0.692447648 | −0.000113467 |
| portfel wybrany probe (downstream) | 20261021 | 20261103 | 0.692493228 | 0.692426931 | +0.000066297 |
| portfel wybrany probe (downstream) | 20261022 | 20261101 | 0.692359019 | 0.692408164 | −0.000049145 |
| portfel wybrany probe (downstream) | 20261022 | 20261102 | 0.692334181 | 0.692427347 | −0.000093166 |
| portfel wybrany probe (downstream) | 20261022 | 20261103 | 0.692493228 | 0.692431250 | +0.000061978 |

| Wariant | Średnia Δ LL | SD populacyjne 9 par | Min | Max | Dodatnie pary |
|---|---:|---:|---:|---:|---:|
| single/ring | +0.000078732 | 0.000039060 | +0.000020068 | +0.000124235 | 9/9 |
| portfel wybrany probe (downstream) | −0.000046807 | 0.000079616 | −0.000152455 | +0.000066297 | 3/9 |

Zmieniano **oba seedy w pełnym układzie krzyżowym 3×3**. Pozwala to pokazać średnie warunkowe po seedzie reservoiru oraz po seedzie LightGBM: dla zwycięzcy każda z trzech średnich reservoiru i każda z trzech średnich LightGBM była dodatnia; dla portfela średnia po każdym seedzie reservoiru była ujemna. Trzy poziomy każdego czynnika i jeden wynik na komórkę nie pozwalają oszacować szerokiej populacyjnej stabilności ani oddzielić interakcji seedów od pozostałego szumu.

### Rzeczywiste pokrycie rodzin i topologii

Kod `ESNConfig` obsługuje `single`, `parallel`, `deep` oraz topologie `ring`, `sparse`, `dense`. Konfiguracja R4 przeszukiwała wyłącznie `ring` i `sparse`; `dense` nie wygenerowano ani nie oceniono downstream w R4. Dense pojawiał się tylko w osobnym, historycznym R3 searchu 48 triali, którego budżety i rozkład prób nie były wyrównane.

W R4 oceniono downstream sześciu liderów probe rodzina×topologia i trzech finalistów grup wejściowych. Każda z sześciu par dostała przynajmniej jeden kandydat LightGBM; trzy pary mają po dwóch kandydatów, bo doszedł finalist input-group. „Najlepszy” oznacza tu najwyższą średnią z dziewięciu wyników w tej wyselekcjonowanej puli.

| Rodzina × topologia | Screening scored / rejected / failed | Downstream kandydaci | Najlepsza downstream Δ LL (trial) | Dodatnie pary; SD populacyjne | Czas triali screeningu | Generowanie cech finalistów |
|---|---:|---:|---:|---:|---:|---:|
| single × ring | 4 / 1 / 1 | 1 | +0.000078732 (arch t61) | 9/9; 0.000039060 | 228.0 s | 0.004 s |
| single × sparse | 3 / 0 / 1 | 2 | +0.000066810 (arch t31) | 6/9; 0.000084117 | 353.9 s | 163.119 s |
| parallel × ring | 6 / 0 / 1 | 2 | +0.000001196 (arch t53) | 6/9; 0.000057389 | 271.5 s | 25.990 s |
| parallel × sparse | 4 / 0 / 0 | 1 | +0.000032020 (arch t9) | 5/9; 0.000078448 | 364.0 s | 0.005 s |
| deep × ring | 28 / 0 / 0 | 2 | +0.000068828 (arch t48) | 7/9; 0.000100812 | 1142.4 s | 35.180 s |
| deep × sparse | 24 / 5 / 0 | 1 | +0.000041810 (arch t34) | 6/9; 0.000082071 | 3073.2 s | 0.004 s |

Łącznie study architektury miało 78 prób: 69 ocenionych probe, 6 odrzuconych i 3 nieudane; wszystkie wymagane 6 kotwic rodzina×topologia ostatecznie ukończono. Próby screeningu były nierówne: `deep/ring` i `deep/sparse` stanowią 52 z 69 ocenionych konfiguracji, wobec 7 single i 10 parallel. Sześć kotwic architektury wyrównywało wejścia do pełnego zestawu i budżet do 128 jednostek; nie wyrównywało natomiast liczby późniejszych prób Optuny.

„18/18 kotwic” oznacza inną rzecz: po jednej konfiguracji dla każdej kombinacji trzech zamrożonych baz rodzinnych i sześciu jednokrotnych grup wejściowych (`returns`, `candle`, `range_volatility`, `volume`, `basis`, `time`). Wszystkie 18 zostały ocenione probe. Bazy były `deep/ring 128u`, `parallel/ring 128u` i `single/sparse 256u`; zatem nazwy grup wejściowych były porównywalne, ale budżet jednostek, topologia i pozostałe parametry baz nie były wspólne. To kontroluje wybór grup warunkowo względem wybranej bazy, nie stanowi wyrównanego porównania rodzin.

### Probe a downstream: ranking i wybór portfela

W obu miejscach dodatnie Δ oznacza `baseline log-loss − augmented log-loss`. Okres i target były te same; w probe i downstream zapisano po 52 120 wierszy walidacyjnych, a w downstream każdy kandydat używał tej samej wspólnej maski. Baseline był jednak inny: probe porównywał logistyczną korektę stałych logitów BASE70 (raw-logit LightGBM seed `20261015`) z tą samą korektą rozszerzoną o PCA; kontrola i wariant probe używały `C=0.1` i wspólnych wierszy correction-fit. Downstream ponownie dopasowywał BASE70 LightGBM dla każdego seedu `20261101–20261103` i dołączał cechy ESN. Probe średniował trzy delty reservoiru, downstream dziewięć sparowanych delt 3×3. Liczb nie należy porównywać jako przyrostów względem identycznego baseline’u.

Ranking probe kontra średnia downstream dla dziewięciu finalistów ma korelację Spearmana `ρ = −0.4667` (`n=9`). To wyselekcjonowana pula, a korelacja jest opisowa, nie niezależny test rankera.

| Kandydat | Probe rank (Δ LL) | Downstream rank (Δ LL) |
|---|---:|---:|
| single/sparse 256u, `range_volatility` (`06b79653…`) | 1 (+0.000089797) | 6 (+0.000011029) |
| deep/ring 128u, `basis` (`80bc3eab…`) | 2 (+0.000078527) | 7 (+0.000010110) |
| parallel/ring 128u, `range_volatility+basis` (`61a793c8…`) | 3 (+0.000056151) | 9 (−0.000007840) |
| deep/ring 128u, pełne wejścia (`b656b64a…`) | 4 (−0.000006147) | 2 (+0.000068828) |
| deep/sparse 256u, pełne wejścia (`9d1bec12…`) | 5 (−0.000026274) | 4 (+0.000041810) |
| parallel/ring 128u, pełne wejścia (`4785383d…`) | 6 (−0.000074351) | 8 (+0.000001196) |
| single/sparse 256u, pełne wejścia (`fba0a1f5…`) | 7 (−0.000111475) | 3 (+0.000066810) |
| single/ring 128u, pełne wejścia (`44e399a1…`) | 8 (−0.000167979) | 1 (+0.000078732) |
| parallel/sparse 256u, pełne wejścia (`dcd078fc…`) | 9 (−0.000180747) | 5 (+0.000032020) |

Przebieg selekcji portfela był w całości probe-only:

| Runda | Skład przed | Decyzja probe | Wynik przed → po | Kryterium |
|---|---|---|---:|---|
| 0 | pusty | wybrano `06b79653…` single/sparse 256u, `range_volatility` | 0 → +0.000089797 | Najlepszy pojedynczy wynik spełniający dodatnią średnią i dodatnie delty w co najmniej 2/3 seedów; tutaj 3/3 |
| 1 | `06b79653…` | wybrano `80bc3eab…` deep/ring 128u, `basis` | +0.000089797 → +0.000158274; margines +0.000068477 | Dodatni margines i wynik łączny, dodatnie delty w co najmniej 2/3 seedów; był to najlepszy kwalifikujący dodatek |
| 2 | oba powyższe | stop; żaden z siedmiu pozostałych dodatków nie miał dodatniego marginesu | +0.000158274 → bez zmiany | Kryterium dodatniego marginesu probe |

Zatem `+0.000158274` jest średnią probe log-loss baseline kontra probe rozszerzony dla trzech seedów reservoiru, przy 52 972 wierszach correction-fit i 52 120 walidacji. Downstream LightGBM ocenił 9 modeli pojedynczych oraz **jeden zapisany portfel** na 9 parach seedów. Ta ocena wykazała, że najlepszy pojedynczy model jest lepszy od portfela na wspólnych danych i seedach; nie zmieniła jednak wyniku funkcji selekcji ani zapisanego pola `selected_candidate_ids`.

Z artefaktów ustalono rozbieżność rankingów i różnice baseline’ów. Nie ma dowodu pozwalającego przypisać ujemny wynik portfela przeuczeniu, liczbie cech, kompresji PCA ani konkretnemu mechanizmowi przyczynowemu.

### Wpływ 22 błędnych prób wejściowych

Pierwszy study `btc_esn_architecture_audit_20261020_input_groups` wykonał 22 próby (20 scored, 2 odrzucone); zapisane efektywne konfiguracje wszystkich 22 miały pełne sześć grup, więc nie testowały wyboru grup. Zostały zachowane w `input_search_search_uncorrected.json` i wykluczone z końcowego zestawu kandydatów.

Korekta użyła nowej nazwy study `btc_esn_architecture_audit_20261020_input_groups_corrected_20261021` oraz świeżego stanu TPE z skonfigurowanym seedem `20261021`; nie załadowała obserwacji starego study. Użyła trzech baz wybranych z wcześniejszego, poprawnego etapu architektury, nie z 22 błędnych triali. Sprawdzenie zapisanych efektywnych konfiguracji potwierdza **59/59** zgodności `input_groups → selected_columns`. Cache key jest hashem całego efektywnego rekordu konfiguracji, zawierającego i grupy, i kolumny, z osobnym identity oraz seedem; różny zestaw wejść nie dziedziczy zatem cache po pełnym wejściu.

Błędne próby zużyły 1887.9 s wspólnego limitu 14 400 s. Nie przeniosły się jednak do samplera ani nie ograniczyły kompletności: study korekcyjne ukończyło 59/59 prób i 18/18 kotwic, downstream zakończył się, a cały zapisany koszt wyniósł 10 850.8 s. Wpływ jest więc potwierdzony jako dodatkowy koszt obliczeń, bez widocznego ucięcia poprawionego etapu lub dowodu skażenia jego rankingów.

### Diagnostyka granic parametrów — ograniczenie i sprostowanie

Diagnostyka w manifeście pochodzi z **rankingu probe architektury**, nie z rankingu LightGBM. Kod porządkuje 69 scored triali malejąco po `mean_delta_logloss`, wybiera `ceil(10%) = 7`; connectivity liczy osobno w 31 sparse trialach i ich top 4. Zakres „blisko granicy” to zewnętrzne 5% **liniowej szerokości** skonfigurowanego przedziału; `suggest_float` nie ustawia `log=True`.

Top probe triale `48, 64, 74, 68, 77, 75, 76` to wszystkie `deep/ring`; żaden nie jest kotwicą. Top sparse triale `34, 57, 58, 14` to wszystkie `deep/sparse`; również nie są kotwicami. Nierówny udział rodzin nie tworzy samych wartości granicznych, ale powoduje, że presja jest warunkowa względem DeepESN, a nie wspólna dla wszystkich rodzin.

Opisowy odczyt dziewięciu finalistów downstream daje 5/9 przy dolnej granicy leak rate, 1/9 przy górnej spectral radius i 1/4 sparse przy dolnej connectivity. To mała pula wybrana probe’em, nie odrębna diagnostyka optimum. Sam zwycięzca ma leak `0.1288` (poza dolnym pasmem), spectral radius `1.1513` (w górnym paśmie), a top downstream sparse `single/sparse` ma connectivity `0.4198` (poza dolnym pasmem). Nie uzasadnia to rozszerzenia zakresu connectivity; pojedynczy zwycięzca przy górnym spectral radius nie wystarcza do rozszerzenia tego zakresu.

Istniejące pomiary pozwalają ocenić warm-up dla testowanego dolnego końca leak rate: top 7 probe triali miało leak `0.0504–0.0839`, wszystkie trzy seedy stabilizowały się w **193–499 minutach** przy zamrożonym kryterium. Nie testowano leak poniżej `0.05`, więc warm-up dla wolniejszej dynamiki poniżej tego zakresu pozostaje nieznany. Wartości historycznej tabeli granic pozostają zachowane, lecz jej wcześniejsza sugestia automatycznego poszerzenia leak/connectivity jest zastąpiona powyższym, warunkowym wnioskiem.

### Jednoznaczne odpowiedzi i następny krok

1. Downstream wygrał `single/ring`, 1×128, study architecture trial 61, kandydat `44e399a127401eb4e7ff0fa6`, pełne wejścia; średnia Δ LL `+0.000078732`.
2. Tak: DeepESN i parallel ESN mają downstream kandydatów dla ring i sparse w R4, po jednym liderze probe na każdą parę rodzina×topologia; nie testowano dense downstream w R4.
3. Tak, seedy reservoiru zmieniano i krzyżowano z trzema seedami LightGBM; stabilność jest opisowa dla 3×3, nie szeroka.
4. Nie: selekcja portfela używała przyrostu probe. Downstream oceniono tylko dla finalistów i jednego wybranego probe portfela.
5. Tak, najlepszy pojedynczy wariant i portfel są bezpośrednio porównywalne na tych samych 52 120 wierszach, dziewięciu parach seedów i baseline’ach dla seedów LightGBM. To nadal porównanie selekcyjne na `search_validation`.
6. Nie widać, by błąd wejść sterował poprawionym samplerem, kandydatami lub cache; zużył 1887.9 s, lecz wszystkie poprawione kotwice i downstream zostały ukończone w ramach limitu.

Rekomendacja: **zamknąć obecny etap bez dalszych obliczeń**. Zachować `44e399a127401eb4e7ff0fa6` jako prowizorycznego kandydata do odrębnie zdefiniowanego potwierdzenia, a probe-owy portfel odrzucić jako zwycięzcę downstream. Kolejna analiza tej samej walidacji pogłębiłaby selekcję na tym samym okresie; diagnostyka granic jest dodatkowo skoncentrowana na DeepESN. Koszt obliczeniowy zamknięcia tego etapu wynosi **0 s nowych stanów i fitów**. Jako punkt odniesienia, poprawiona końcowa ocena 10 wariantów (9 single + jeden portfel) kosztowała 499.318 s, w tym 224.302 s generowania cech i 269.663 s dla 93 fitów LightGBM. Nie estymuję niezależnego confirmation runtime na podstawie chronionych artefaktów.

### Maszynowo czytelne tabele

- [Porównanie rodzin i topologii](btc_esn_architecture_audit_20261020/r4_family_topology_comparison.csv)
- [Wszystkie pary seedów zwycięzcy i portfela](btc_esn_architecture_audit_20261020/r4_winner_portfolio_seed_results.csv)
- [Rankingi probe i LightGBM finalistów](btc_esn_architecture_audit_20261020/r4_candidate_rankings.csv)
- [Pełny ślad selekcji portfela probe](btc_esn_architecture_audit_20261020/r4_portfolio_probe_selection_trace.csv)
- [Podsumowanie interpretacji i metadane](btc_esn_architecture_audit_20261020/r4_interpretation_summary.json)
