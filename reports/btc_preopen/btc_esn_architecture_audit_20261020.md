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

Leak rate wykazuje silną presję dolnej granicy (7/7 najlepszych triali w dolnych 5% zakresu); sparse connectivity również sugeruje sprawdzenie niższych wartości (2/4 własnego top decile). Dla spectral radius presja górna jest słabsza (1/7 top decile). W następnym study warto rozszerzyć dolny zakres leak rate i connectivity oraz umiarkowanie sprawdzić spectral radius powyżej 1.2. Input scaling i bias scaling nie wykazały presji w top decile.

Poprawiony etap można wznowić poleceniem `python run_btc_esn_input_group_correction_20261021.py`; pełny runner architektury: `python run_btc_esn_architecture_audit_20261020.py`.

## Zakres wyników

Wszystkie R4 probe i downstream wyniki pozostają selekcyjne dla `search_validation`. Nie są niezależnym potwierdzeniem. Holdout i późniejsze dane/etykiety nie zostały odczytane.
