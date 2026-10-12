# BTC i rynek: zagnieżdżony test korekt prawdopodobieństwa

> Nowy eksperyment selekcji z 12 października 2026. Zmiana procedury wyboru modeli i polityki; nie poprawka księgowania.
> Holdout lipiec–październik był wcześniej oglądany. Jego wyniki są historyczną diagnostyką, nie niezależnym potwierdzeniem.

## Werdykt

Nested May-June OOF (8,676 markets) ranks A_BTC_calibrated first by log-loss (0.69231). E-B=-0.00057 [-0.00124, +0.00013], E-D=-0.00045 [-0.00106, +0.00018]. Frozen policy: A_BTC_calibrated_ev_1.0000%. July-October is historical diagnostics only.

## Zamrożona procedura i pokrycie

Źródło wejściowe zawiera 80,352 zapisanych predykcji BTC; eksperyment użył 8,676 wspólnych rynków OOF w rozwoju i 24,736 rynków holdout. Wspólna próba wymaga oficjalnego wyniku, przyczynowej skalibrowanej prognozy BTC oraz poprawnych obu stron T−60. T−59 służy wyłącznie do wyceny wykonania.

Jedynym wejściem BTC dla E/F jest istniejące skalibrowane p_BTC. Nie trenowano ani nie refitowano głównego modelu BTC. E używa `p=(1-w)*p_market+w*p_BTC` z w=[0, 0.1, 0.25, 0.5, 1]. F używa offsetu rynkowego `logit(p)=logit(p_market)+a+b*(logit(p_BTC)-logit(p_market))`, z `lambda*(a²+b²)/2` dodanym do średniej log-loss; lambda=[0.1, 1, 10]. Obie rodziny zawierają dokładną opcję bez korekty równą p_market.

C i D pozostały kontrolami. W zewnętrznym foldu majowym parametry dopasowano tylko na wcześniejszych prefiksach kwietnia; w czerwcowym tylko na wcześniejszych prefiksach do maja. W każdym wewnętrznym foldzie trening odcina etykiety z `label_available_at_utc` późniejszym niż początek walidacji. OOF obejmuje wyłącznie maj i czerwiec.

| Zewnętrzny fold | Prefiks treningowy | Walidacja wewnętrzna | Rynki outer train | Rynki outer valid |
|---|---|---|---:|---:|
| dev_2026q2_b | < 2026-05-01T00:00:00Z z purge etykiet | 2026-04-22T00:00:00Z–2026-04-27T00:00:00Z (649); 2026-04-27T00:00:00Z–2026-05-01T00:00:00Z (519) | 2,119 | 5,253 |
| dev_2026q2_c | < 2026-06-01T00:00:00Z z purge etykiet | 2026-05-01T00:00:00Z–2026-05-16T00:00:00Z (2,166); 2026-05-16T00:00:00Z–2026-06-01T00:00:00Z (3,087) | 7,371 | 3,423 |

Dobór końcowych parametrów do zamrożonej oceny holdout używa odrębnie zadeklarowanych foldów majowym i czerwcowym, wyłącznie przed cutoffem 1 lipca. Wyników tych foldów nie użyto do doboru parametrów predykcji outer OOF z tych samych miesięcy.

### Parametry wybrane dla końcowych modeli holdout

| Rodzina | Wariant | Parametr | Wartość | Log-loss wewnętrzny | n walidacji |
|---|---|---|---:|---:|---:|
| C | C_C_10 | C | 10.0000 | 0.69356 | 8,676 |
| D | D_C_10 | C | 10.0000 | 0.69286 | 8,676 |
| E | E_w_1 | w | 1.0000 | 0.69231 | 8,676 |
| F | F_no_correction | no_correction | n/a | 0.69299 | 8,676 |

## Prawdopodobieństwa na tych samych rynkach

Dla C–F development prognozy pochodzą wyłącznie z outer OOF. Holdout pokazuje modele z parametrami zamrożonymi przed jego oceną. Ujemna sparowana różnica log-loss/Brier sprzyja pierwszemu modelowi; bootstrap używa wspólnych ruchomych bloków 3 dni UTC.

| Okres | Model | n | Log-loss | Brier | AUC | Intercept kalibracji | Slope | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| development_nested_oof | A_BTC_calibrated | 8,676 | 0.69231 | 0.24958 | 0.5251 | -0.049 | 1.106 | 0.0224 |
| development_nested_oof | B_market_reference | 8,676 | 0.69299 | 0.24992 | 0.5157 | -0.045 | 0.747 | 0.0135 |
| development_nested_oof | C_market_logistic | 8,676 | 0.69379 | 0.25032 | 0.5219 | -0.163 | 2.047 | 0.0271 |
| development_nested_oof | D_market_plus_BTC | 8,676 | 0.69287 | 0.24986 | 0.5288 | -0.096 | 1.022 | 0.0238 |
| development_nested_oof | E_probability_mixture | 8,676 | 0.69242 | 0.24963 | 0.5241 | -0.049 | 1.185 | 0.0181 |
| development_nested_oof | F_regularized_logit_correction | 8,676 | 0.69322 | 0.25003 | 0.5241 | -0.072 | 0.865 | 0.0206 |
| holdout_diagnostic | A_BTC_calibrated | 24,736 | 0.69245 | 0.24965 | 0.5227 | -0.008 | 0.800 | 0.0091 |
| holdout_diagnostic | B_market_reference | 24,736 | 0.69158 | 0.24923 | 0.5202 | -0.012 | 1.053 | 0.0084 |
| holdout_diagnostic | C_market_logistic | 24,736 | 0.69171 | 0.24929 | 0.5202 | 0.020 | 1.397 | 0.0066 |
| holdout_diagnostic | D_market_plus_BTC | 24,736 | 0.69176 | 0.24931 | 0.5289 | 0.019 | 0.896 | 0.0123 |
| holdout_diagnostic | E_probability_mixture | 24,736 | 0.69245 | 0.24965 | 0.5227 | -0.008 | 0.800 | 0.0094 |
| holdout_diagnostic | F_regularized_logit_correction | 24,736 | 0.69158 | 0.24923 | 0.5202 | -0.012 | 1.053 | 0.0099 |

### Jakość OOF w zewnętrznych foldach

| Fold | Model | n | Log-loss | Brier | Slope | ECE |
|---|---|---:|---:|---:|---:|---:|
| dev_2026q2_b | A_BTC_calibrated | 5,253 | 0.69238 | 0.24962 | 1.052 | 0.0185 |
| dev_2026q2_c | A_BTC_calibrated | 3,423 | 0.69221 | 0.24953 | 1.203 | 0.0310 |
| dev_2026q2_b | B_market_reference | 5,253 | 0.69282 | 0.24983 | 0.715 | 0.0126 |
| dev_2026q2_c | B_market_reference | 3,423 | 0.69325 | 0.25004 | 0.692 | 0.0296 |
| dev_2026q2_b | C_market_logistic | 5,253 | 0.69380 | 0.25033 | 0.001 | 0.0228 |
| dev_2026q2_c | C_market_logistic | 3,423 | 0.69377 | 0.25031 | 2.103 | 0.0338 |
| dev_2026q2_b | D_market_plus_BTC | 5,253 | 0.69298 | 0.24991 | 0.872 | 0.0278 |
| dev_2026q2_c | D_market_plus_BTC | 3,423 | 0.69270 | 0.24978 | 1.050 | 0.0324 |
| dev_2026q2_b | E_probability_mixture | 5,253 | 0.69238 | 0.24962 | 1.052 | 0.0197 |
| dev_2026q2_c | E_probability_mixture | 3,423 | 0.69247 | 0.24966 | 1.454 | 0.0296 |
| dev_2026q2_b | F_regularized_logit_correction | 5,253 | 0.69320 | 0.25002 | 0.733 | 0.0170 |
| dev_2026q2_c | F_regularized_logit_correction | 3,423 | 0.69325 | 0.25004 | 0.692 | 0.0311 |

| Okres | Porównanie | Δ log-loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---|---:|---:|---:|
| development_nested_oof | E-B | -0.00057 [-0.00124, +0.00013] | -0.00028 [-0.00062, +0.00006] | +0.0084 [-0.0047, +0.0214] |
| development_nested_oof | E-D | -0.00045 [-0.00106, +0.00018] | -0.00022 [-0.00053, +0.00009] | -0.0046 [-0.0082, +0.0004] |
| development_nested_oof | F-B | +0.00023 [-0.00011, +0.00060] | +0.00011 [-0.00006, +0.00030] | +0.0084 [-0.0030, +0.0176] |
| development_nested_oof | F-D | +0.00035 [-0.00057, +0.00128] | +0.00017 [-0.00029, +0.00064] | -0.0046 [-0.0164, +0.0072] |
| development_nested_oof | D-B | -0.00012 [-0.00119, +0.00102] | -0.00006 [-0.00059, +0.00051] | +0.0130 [-0.0003, +0.0261] |
| development_nested_oof | C-B | +0.00080 [+0.00018, +0.00153] | +0.00040 [+0.00009, +0.00077] | +0.0061 [-0.0072, +0.0163] |
| development_nested_oof | D-C | -0.00092 [-0.00189, +0.00010] | -0.00046 [-0.00094, +0.00005] | +0.0069 [-0.0058, +0.0210] |
| development_nested_oof | B-A | +0.00067 [-0.00013, +0.00147] | +0.00033 [-0.00006, +0.00073] | -0.0094 [-0.0218, +0.0048] |
| holdout_diagnostic | E-B | +0.00087 [-0.00018, +0.00216] | +0.00042 [-0.00010, +0.00104] | +0.0026 [-0.0038, +0.0102] |
| holdout_diagnostic | E-D | +0.00069 [+0.00013, +0.00142] | +0.00034 [+0.00006, +0.00071] | -0.0062 [-0.0117, -0.0021] |
| holdout_diagnostic | F-B | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.0000 [+0.0000, +0.0000] |
| holdout_diagnostic | F-D | -0.00018 [-0.00082, +0.00052] | -0.00008 [-0.00039, +0.00027] | -0.0087 [-0.0144, -0.0044] |
| holdout_diagnostic | D-B | +0.00018 [-0.00052, +0.00082] | +0.00008 [-0.00027, +0.00039] | +0.0087 [+0.0044, +0.0144] |
| holdout_diagnostic | C-B | +0.00013 [-0.00007, +0.00038] | +0.00006 [-0.00004, +0.00018] | -0.0000 [-0.0000, +0.0000] |
| holdout_diagnostic | D-C | +0.00005 [-0.00062, +0.00060] | +0.00002 [-0.00031, +0.00029] | +0.0087 [+0.0044, +0.0144] |
| holdout_diagnostic | B-A | -0.00087 [-0.00216, +0.00018] | -0.00042 [-0.00104, +0.00010] | -0.0026 [-0.0102, +0.0038] |

### Jakość prawdopodobieństw holdoutu według miesiąca

| Miesiąc UTC | Model | n | Log-loss | Brier | AUC |
|---|---|---:|---:|---:|---:|
| 2026-07 | A_BTC_calibrated | 6,795 | 0.69336 | 0.25010 | 0.5146 |
| 2026-08 | A_BTC_calibrated | 7,576 | 0.69203 | 0.24944 | 0.5263 |
| 2026-09 | A_BTC_calibrated | 8,639 | 0.69187 | 0.24936 | 0.5290 |
| 2026-10 | A_BTC_calibrated | 1,726 | 0.69355 | 0.25020 | 0.5095 |
| 2026-07 | B_market_reference | 6,795 | 0.69121 | 0.24906 | 0.5116 |
| 2026-08 | B_market_reference | 7,576 | 0.69136 | 0.24913 | 0.5177 |
| 2026-09 | B_market_reference | 8,639 | 0.69211 | 0.24948 | 0.5266 |
| 2026-10 | B_market_reference | 1,726 | 0.69132 | 0.24909 | 0.5327 |
| 2026-07 | C_market_logistic | 6,795 | 0.69139 | 0.24914 | 0.5116 |
| 2026-08 | C_market_logistic | 7,576 | 0.69135 | 0.24912 | 0.5177 |
| 2026-09 | C_market_logistic | 8,639 | 0.69220 | 0.24953 | 0.5266 |
| 2026-10 | C_market_logistic | 1,726 | 0.69214 | 0.24950 | 0.5327 |
| 2026-07 | D_market_plus_BTC | 6,795 | 0.69242 | 0.24964 | 0.5244 |
| 2026-08 | D_market_plus_BTC | 7,576 | 0.69088 | 0.24887 | 0.5338 |
| 2026-09 | D_market_plus_BTC | 8,639 | 0.69163 | 0.24924 | 0.5309 |
| 2026-10 | D_market_plus_BTC | 1,726 | 0.69368 | 0.25027 | 0.5178 |
| 2026-07 | E_probability_mixture | 6,795 | 0.69336 | 0.25010 | 0.5146 |
| 2026-08 | E_probability_mixture | 7,576 | 0.69203 | 0.24944 | 0.5263 |
| 2026-09 | E_probability_mixture | 8,639 | 0.69187 | 0.24936 | 0.5290 |
| 2026-10 | E_probability_mixture | 1,726 | 0.69355 | 0.25020 | 0.5095 |
| 2026-07 | F_regularized_logit_correction | 6,795 | 0.69121 | 0.24906 | 0.5116 |
| 2026-08 | F_regularized_logit_correction | 7,576 | 0.69136 | 0.24913 | 0.5177 |
| 2026-09 | F_regularized_logit_correction | 8,639 | 0.69211 | 0.24948 | 0.5266 |
| 2026-10 | F_regularized_logit_correction | 1,726 | 0.69132 | 0.24909 | 0.5327 |

## OOF zlecenia niezależne po $5

Wszystkie sygnały używają strony wybranej z T−60 i rzeczywistego T−59 do wykonania. Zlecenia nie dzielą gotówki. Próg to dotychczasowa siatka 0%, 0.25%, 0.5%, 1% i 2%. Przedziały są blokowym bootstrapem 3-dniowym dla łącznego PnL po fee. Koncentracja pokazuje udział trzech dni o największym bezwzględnym PnL.

| Model | Próg EV | Sygnały | Fille | Aktywne dni | Fee eq. | Oczekiwane EV T−59 | PnL netto | CI 95% PnL | Net/obrót | Top 3 dni | Maj / czerwiec PnL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_BTC_calibrated | 0.00% | 1,621 | 1,606 | 44 | $286.48 | $254.21 | $200.77 | [$-106.52, $544.19] | 2.50% | 2.5% | $33.20 / $167.58 |
| A_BTC_calibrated | 0.25% | 1,456 | 1,443 | 44 | $257.87 | $253.50 | $213.01 | [$-69.72, $551.03] | 2.95% | 2.8% | $15.63 / $197.38 |
| A_BTC_calibrated | 0.50% | 1,304 | 1,292 | 44 | $231.48 | $250.70 | $246.83 | [$-31.71, $563.08] | 3.82% | 2.8% | $51.79 / $195.04 |
| A_BTC_calibrated | 1.00% | 1,044 | 1,035 | 43 | $186.51 | $241.51 | $264.52 | [$37.74, $537.86] | 5.11% | 3.0% | $87.53 / $176.99 |
| A_BTC_calibrated | 2.00% | 649 | 640 | 43 | $117.08 | $212.51 | $161.63 | [$4.34, $329.17] | 5.05% | 3.0% | $95.45 / $66.18 |
| B_market_reference | 0.00% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 | [$0.00, $0.00] | n/a% | 0.0% | $0.00 / $0.00 |
| B_market_reference | 0.25% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 | [$0.00, $0.00] | n/a% | 0.0% | $0.00 / $0.00 |
| B_market_reference | 0.50% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 | [$0.00, $0.00] | n/a% | 0.0% | $0.00 / $0.00 |
| B_market_reference | 1.00% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 | [$0.00, $0.00] | n/a% | 0.0% | $0.00 / $0.00 |
| B_market_reference | 2.00% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 | [$0.00, $0.00] | n/a% | 0.0% | $0.00 / $0.00 |
| C_market_logistic | 0.00% | 1,117 | 1,109 | 42 | $203.24 | $210.90 | $-140.99 | [$-394.92, $62.59] | -2.54% | 2.1% | $-132.39 / $-8.60 |
| C_market_logistic | 0.25% | 1,019 | 1,011 | 42 | $185.39 | $210.04 | $-195.87 | [$-452.12, $33.31] | -3.87% | 2.3% | $-132.39 / $-63.48 |
| C_market_logistic | 0.50% | 986 | 979 | 42 | $179.36 | $209.44 | $-245.39 | [$-505.38, $-12.19] | -5.01% | 2.4% | $-141.93 / $-103.46 |
| C_market_logistic | 1.00% | 505 | 501 | 36 | $95.23 | $191.50 | $-123.41 | [$-292.72, $60.15] | -4.93% | 4.2% | $-13.77 / $-109.64 |
| C_market_logistic | 2.00% | 424 | 420 | 36 | $80.09 | $184.85 | $-3.45 | [$-163.62, $175.49] | -0.16% | 4.4% | $-13.94 / $10.49 |
| D_market_plus_BTC | 0.00% | 2,042 | 2,015 | 45 | $351.56 | $303.19 | $9.63 | [$-431.53, $443.44] | 0.10% | 2.7% | $7.75 / $1.89 |
| D_market_plus_BTC | 0.25% | 1,881 | 1,856 | 45 | $324.29 | $302.10 | $35.18 | [$-363.65, $447.50] | 0.38% | 2.7% | $-13.79 / $48.97 |
| D_market_plus_BTC | 0.50% | 1,722 | 1,699 | 45 | $297.15 | $299.83 | $155.25 | [$-190.81, $546.47] | 1.83% | 2.9% | $21.55 / $133.71 |
| D_market_plus_BTC | 1.00% | 1,433 | 1,413 | 45 | $247.74 | $289.65 | $194.29 | [$-111.55, $517.54] | 2.75% | 2.9% | $9.81 / $184.48 |
| D_market_plus_BTC | 2.00% | 961 | 949 | 43 | $167.61 | $255.87 | $88.89 | [$-136.37, $313.33] | 1.87% | 3.2% | $25.59 / $63.30 |
| E_probability_mixture | 0.00% | 789 | 779 | 42 | $141.77 | $155.09 | $53.25 | [$-180.22, $300.29] | 1.37% | 4.3% | $33.20 / $20.05 |
| E_probability_mixture | 0.25% | 684 | 676 | 42 | $123.46 | $154.70 | $48.97 | [$-135.25, $269.22] | 1.45% | 4.9% | $15.63 / $33.35 |
| E_probability_mixture | 0.50% | 597 | 590 | 41 | $108.27 | $153.13 | $61.41 | [$-107.58, $249.88] | 2.08% | 4.7% | $51.79 / $9.62 |
| E_probability_mixture | 1.00% | 467 | 462 | 38 | $85.64 | $148.42 | $102.16 | [$-53.48, $286.62] | 4.42% | 5.3% | $87.53 / $14.64 |
| E_probability_mixture | 2.00% | 290 | 285 | 37 | $54.02 | $135.57 | $126.88 | [$-10.01, $289.30] | 8.90% | 6.9% | $95.45 / $31.43 |
| F_regularized_logit_correction | 0.00% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 | [$0.00, $0.00] | n/a% | 0.0% | $0.00 / $0.00 |
| F_regularized_logit_correction | 0.25% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 | [$0.00, $0.00] | n/a% | 0.0% | $0.00 / $0.00 |
| F_regularized_logit_correction | 0.50% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 | [$0.00, $0.00] | n/a% | 0.0% | $0.00 / $0.00 |
| F_regularized_logit_correction | 1.00% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 | [$0.00, $0.00] | n/a% | 0.0% | $0.00 / $0.00 |
| F_regularized_logit_correction | 2.00% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 | [$0.00, $0.00] | n/a% | 0.0% | $0.00 / $0.00 |

Kryterium aktywnego wyboru było zadeklarowane przed wynikiem: co najmniej 50 filli, 15 aktywnych dni, co najmniej 10 filli i dodatni PnL w każdym z dwóch foldów oraz dolna granica 95% blokowego CI całkowitego PnL powyżej zera. Przedziały dotyczą stałego kandydata i nie korygują wielokrotnego porównania modeli/progów.

Zamrożona polityka: `A_BTC_calibrated_ev_1.0000%`. Passed the predeclared active-policy gate.
Kontrola no-trade: 0 fille, PnL $0.00, CI [$0.00, $0.00].

## Prognozy na tych samych wykonaniach

W każdym koszyku ustalono rynek i stronę przez A albo D, a następnie zachowano te same fille T−59, fee, liczbę udziałów, debet i wypłatę. Zmieniają się tylko prognozowane p(side), log-loss/Brier i oczekiwane EV.

| Okres | Koszyk wybrany przez | Prognoza | Fille | Log-loss | Brier | Śr. p(side) / win rate | Oczekiwane EV T−59 | Zrealizowany PnL | EV−PnL |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| development_oof | A_BTC_calibrated | A_BTC_calibrated | 1,606 | 0.69047 | 0.24866 | 0.522 / 0.521 | $254.21 | $200.77 | $53.44 [$-258.03, $344.85] |
| development_oof | A_BTC_calibrated | B_market_reference | 1,606 | 0.69284 | 0.24983 | 0.486 / 0.521 | $-368.82 | $200.77 | $-569.59 [$-1,081.22, $-126.99] |
| development_oof | A_BTC_calibrated | C_market_logistic | 1,606 | 0.69389 | 0.25037 | 0.497 / 0.521 | $-150.71 | $200.77 | $-351.49 [$-783.44, $22.37] |
| development_oof | A_BTC_calibrated | D_market_plus_BTC | 1,606 | 0.69090 | 0.24888 | 0.518 / 0.521 | $173.25 | $200.77 | $-27.53 [$-337.27, $269.41] |
| development_oof | A_BTC_calibrated | E_probability_mixture | 1,606 | 0.69112 | 0.24899 | 0.512 / 0.521 | $94.62 | $200.77 | $-106.16 [$-480.90, $229.66] |
| development_oof | A_BTC_calibrated | F_regularized_logit_correction | 1,606 | 0.69324 | 0.25003 | 0.485 / 0.521 | $-378.82 | $200.77 | $-579.60 [$-1,097.63, $-132.87] |
| development_oof | D_market_plus_BTC | A_BTC_calibrated | 2,015 | 0.69168 | 0.24927 | 0.524 / 0.520 | $117.27 | $9.63 | $107.63 [$-326.54, $570.75] |
| development_oof | D_market_plus_BTC | B_market_reference | 2,015 | 0.69333 | 0.25007 | 0.498 / 0.520 | $-449.30 | $9.63 | $-458.93 [$-928.22, $2.91] |
| development_oof | D_market_plus_BTC | C_market_logistic | 2,015 | 0.69334 | 0.25010 | 0.512 / 0.520 | $-118.97 | $9.63 | $-128.61 [$-635.41, $350.74] |
| development_oof | D_market_plus_BTC | D_market_plus_BTC | 2,015 | 0.69256 | 0.24971 | 0.534 / 0.520 | $303.19 | $9.63 | $293.55 [$-166.61, $784.57] |
| development_oof | D_market_plus_BTC | E_probability_mixture | 2,015 | 0.69195 | 0.24940 | 0.518 / 0.520 | $-6.85 | $9.63 | $-16.49 [$-493.76, $448.71] |
| development_oof | D_market_plus_BTC | F_regularized_logit_correction | 2,015 | 0.69333 | 0.25007 | 0.504 / 0.520 | $-322.41 | $9.63 | $-332.05 [$-812.67, $111.72] |
| holdout_diagnostic | A_BTC_calibrated | A_BTC_calibrated | 5,871 | 0.69375 | 0.25030 | 0.525 / 0.500 | $1,282.12 | $-557.30 | $1,839.43 [$831.50, $2,947.64] |
| holdout_diagnostic | A_BTC_calibrated | B_market_reference | 5,871 | 0.68870 | 0.24784 | 0.484 / 0.500 | $-1,429.37 | $-557.30 | $-872.07 [$-1,616.08, $-142.51] |
| holdout_diagnostic | A_BTC_calibrated | C_market_logistic | 5,871 | 0.68918 | 0.24805 | 0.488 / 0.500 | $-1,109.22 | $-557.30 | $-551.91 [$-1,308.38, $217.82] |
| holdout_diagnostic | A_BTC_calibrated | D_market_plus_BTC | 5,871 | 0.69117 | 0.24902 | 0.518 / 0.500 | $683.97 | $-557.30 | $1,241.28 [$392.94, $2,103.47] |
| holdout_diagnostic | A_BTC_calibrated | E_probability_mixture | 5,871 | 0.69375 | 0.25030 | 0.525 / 0.500 | $1,282.12 | $-557.30 | $1,839.43 [$831.50, $2,947.64] |
| holdout_diagnostic | A_BTC_calibrated | F_regularized_logit_correction | 5,871 | 0.68870 | 0.24784 | 0.484 / 0.500 | $-1,429.37 | $-557.30 | $-872.07 [$-1,616.08, $-142.51] |
| holdout_diagnostic | D_market_plus_BTC | A_BTC_calibrated | 6,615 | 0.69264 | 0.24974 | 0.529 / 0.510 | $1,008.26 | $-635.50 | $1,643.76 [$696.79, $2,700.36] |
| holdout_diagnostic | D_market_plus_BTC | B_market_reference | 6,615 | 0.68828 | 0.24763 | 0.494 / 0.510 | $-1,560.21 | $-635.50 | $-924.71 [$-1,691.02, $-157.31] |
| holdout_diagnostic | D_market_plus_BTC | C_market_logistic | 6,615 | 0.68874 | 0.24783 | 0.499 / 0.510 | $-1,192.93 | $-635.50 | $-557.42 [$-1,324.83, $237.28] |
| holdout_diagnostic | D_market_plus_BTC | D_market_plus_BTC | 6,615 | 0.69017 | 0.24852 | 0.531 / 0.510 | $945.76 | $-635.50 | $1,581.26 [$696.24, $2,472.41] |
| holdout_diagnostic | D_market_plus_BTC | E_probability_mixture | 6,615 | 0.69264 | 0.24974 | 0.529 / 0.510 | $1,008.26 | $-635.50 | $1,643.76 [$696.79, $2,700.36] |
| holdout_diagnostic | D_market_plus_BTC | F_regularized_logit_correction | 6,615 | 0.68828 | 0.24763 | 0.494 / 0.510 | $-1,560.21 | $-635.50 | $-924.71 [$-1,691.02, $-157.31] |

### Jakość własnych selekcji dodatniego EV

| Okres | Model wybierający | Próg EV | Fille | Log-loss side | Brier side | Śr. p(side) / win rate |
|---|---|---:|---:|---:|---:|---:|
| development_oof | A_BTC_calibrated | 0.00% | 1,606 | 0.69047 | 0.24866 | 0.522 / 0.521 |
| development_oof | A_BTC_calibrated | 0.50% | 1,292 | 0.68951 | 0.24819 | 0.523 / 0.525 |
| development_oof | A_BTC_calibrated | 1.00% | 1,035 | 0.68971 | 0.24829 | 0.523 / 0.528 |
| development_oof | B_market_reference | 0.00% | 0 | n/a | n/a | n/a / n/a |
| development_oof | B_market_reference | 0.50% | 0 | n/a | n/a | n/a / n/a |
| development_oof | B_market_reference | 1.00% | 0 | n/a | n/a | n/a / n/a |
| development_oof | C_market_logistic | 0.00% | 1,109 | 0.69408 | 0.25047 | 0.511 / 0.482 |
| development_oof | C_market_logistic | 0.50% | 979 | 0.69399 | 0.25042 | 0.513 / 0.470 |
| development_oof | C_market_logistic | 1.00% | 501 | 0.69330 | 0.25008 | 0.506 / 0.449 |
| development_oof | D_market_plus_BTC | 0.00% | 2,015 | 0.69256 | 0.24971 | 0.534 / 0.520 |
| development_oof | D_market_plus_BTC | 0.50% | 1,699 | 0.69118 | 0.24902 | 0.535 / 0.527 |
| development_oof | D_market_plus_BTC | 1.00% | 1,413 | 0.69062 | 0.24874 | 0.537 / 0.531 |
| development_oof | E_probability_mixture | 0.00% | 779 | 0.69139 | 0.24912 | 0.515 / 0.504 |
| development_oof | E_probability_mixture | 0.50% | 590 | 0.69035 | 0.24860 | 0.515 / 0.503 |
| development_oof | E_probability_mixture | 1.00% | 462 | 0.69032 | 0.24859 | 0.514 / 0.509 |
| development_oof | F_regularized_logit_correction | 0.00% | 0 | n/a | n/a | n/a / n/a |
| development_oof | F_regularized_logit_correction | 0.50% | 0 | n/a | n/a | n/a / n/a |
| development_oof | F_regularized_logit_correction | 1.00% | 0 | n/a | n/a | n/a / n/a |
| holdout_diagnostic | A_BTC_calibrated | 0.00% | 5,871 | 0.69375 | 0.25030 | 0.525 / 0.500 |
| holdout_diagnostic | A_BTC_calibrated | 0.50% | 4,862 | 0.69407 | 0.25046 | 0.526 / 0.492 |
| holdout_diagnostic | A_BTC_calibrated | 1.00% | 4,018 | 0.69436 | 0.25060 | 0.526 / 0.486 |
| holdout_diagnostic | B_market_reference | 0.00% | 0 | n/a | n/a | n/a / n/a |
| holdout_diagnostic | B_market_reference | 0.50% | 0 | n/a | n/a | n/a / n/a |
| holdout_diagnostic | B_market_reference | 1.00% | 0 | n/a | n/a | n/a / n/a |
| holdout_diagnostic | C_market_logistic | 0.00% | 329 | 0.62462 | 0.21644 | 0.385 / 0.310 |
| holdout_diagnostic | C_market_logistic | 0.50% | 285 | 0.61240 | 0.21045 | 0.378 / 0.281 |
| holdout_diagnostic | C_market_logistic | 1.00% | 244 | 0.60587 | 0.20732 | 0.371 / 0.279 |
| holdout_diagnostic | D_market_plus_BTC | 0.00% | 6,615 | 0.69017 | 0.24852 | 0.531 / 0.510 |
| holdout_diagnostic | D_market_plus_BTC | 0.50% | 5,618 | 0.68958 | 0.24823 | 0.531 / 0.510 |
| holdout_diagnostic | D_market_plus_BTC | 1.00% | 4,677 | 0.68954 | 0.24821 | 0.532 / 0.504 |
| holdout_diagnostic | E_probability_mixture | 0.00% | 5,871 | 0.69375 | 0.25030 | 0.525 / 0.500 |
| holdout_diagnostic | E_probability_mixture | 0.50% | 4,862 | 0.69407 | 0.25046 | 0.526 / 0.492 |
| holdout_diagnostic | E_probability_mixture | 1.00% | 4,018 | 0.69436 | 0.25060 | 0.526 / 0.486 |
| holdout_diagnostic | F_regularized_logit_correction | 0.00% | 0 | n/a | n/a | n/a / n/a |
| holdout_diagnostic | F_regularized_logit_correction | 0.50% | 0 | n/a | n/a | n/a / n/a |
| holdout_diagnostic | F_regularized_logit_correction | 1.00% | 0 | n/a | n/a | n/a / n/a |

## Holdout: diagnostyka po zamrożeniu

| Model | Próg EV | Sygnały | Fille | Aktywne dni | Oczekiwane EV T−59 | PnL netto | Fee eq. |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_BTC_calibrated | 0.00% | 6,248 | 5,871 | 88 | $1,282.12 | $-557.30 | $1,047.12 |
| A_BTC_calibrated | 0.25% | 5,705 | 5,358 | 88 | $1,280.36 | $-578.32 | $958.31 |
| A_BTC_calibrated | 0.50% | 5,176 | 4,862 | 88 | $1,272.16 | $-686.83 | $872.20 |
| A_BTC_calibrated | 1.00% | 4,272 | 4,018 | 88 | $1,242.33 | $-732.16 | $725.31 |
| A_BTC_calibrated | 2.00% | 2,872 | 2,707 | 88 | $1,149.97 | $-449.26 | $497.51 |
| B_market_reference | 0.00% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 |
| B_market_reference | 0.25% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 |
| B_market_reference | 0.50% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 |
| B_market_reference | 1.00% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 |
| B_market_reference | 2.00% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 |
| C_market_logistic | 0.00% | 336 | 329 | 55 | $66.19 | $-294.52 | $74.07 |
| C_market_logistic | 0.25% | 322 | 316 | 54 | $66.14 | $-303.69 | $71.31 |
| C_market_logistic | 0.50% | 290 | 285 | 48 | $65.82 | $-330.58 | $65.09 |
| C_market_logistic | 1.00% | 247 | 244 | 37 | $64.61 | $-271.86 | $56.48 |
| C_market_logistic | 2.00% | 209 | 206 | 30 | $61.76 | $-236.71 | $48.35 |
| D_market_plus_BTC | 0.00% | 6,937 | 6,615 | 88 | $945.76 | $-635.50 | $1,156.66 |
| D_market_plus_BTC | 0.25% | 6,404 | 6,106 | 88 | $942.28 | $-539.24 | $1,068.94 |
| D_market_plus_BTC | 0.50% | 5,890 | 5,618 | 87 | $933.85 | $-452.79 | $985.22 |
| D_market_plus_BTC | 1.00% | 4,890 | 4,677 | 87 | $898.72 | $-571.29 | $823.34 |
| D_market_plus_BTC | 2.00% | 3,221 | 3,083 | 86 | $786.74 | $-394.38 | $548.96 |
| E_probability_mixture | 0.00% | 6,248 | 5,871 | 88 | $1,282.12 | $-557.30 | $1,047.12 |
| E_probability_mixture | 0.25% | 5,705 | 5,358 | 88 | $1,280.36 | $-578.32 | $958.31 |
| E_probability_mixture | 0.50% | 5,176 | 4,862 | 88 | $1,272.16 | $-686.83 | $872.20 |
| E_probability_mixture | 1.00% | 4,272 | 4,018 | 88 | $1,242.33 | $-732.16 | $725.31 |
| E_probability_mixture | 2.00% | 2,872 | 2,707 | 88 | $1,149.97 | $-449.26 | $497.51 |
| F_regularized_logit_correction | 0.00% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 |
| F_regularized_logit_correction | 0.25% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 |
| F_regularized_logit_correction | 0.50% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 |
| F_regularized_logit_correction | 1.00% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 |
| F_regularized_logit_correction | 2.00% | 0 | 0 | 0 | $0.00 | $0.00 | $0.00 |

### PnL miesięczny holdout przy hurdle 0%

| Miesiąc UTC | Model | Fille | Aktywne dni | Oczekiwane EV T−59 | PnL netto |
|---|---|---:|---:|---:|---:|
| 2026-07 | A_BTC_calibrated | 1,824 | 25 | $401.78 | $-494.40 |
| 2026-08 | A_BTC_calibrated | 1,788 | 27 | $582.41 | $108.68 |
| 2026-09 | A_BTC_calibrated | 1,755 | 30 | $238.31 | $30.06 |
| 2026-10 | A_BTC_calibrated | 504 | 6 | $59.63 | $-201.64 |
| 2026-07 | B_market_reference | 0 | 0 | $0.00 | $0.00 |
| 2026-08 | B_market_reference | 0 | 0 | $0.00 | $0.00 |
| 2026-09 | B_market_reference | 0 | 0 | $0.00 | $0.00 |
| 2026-10 | B_market_reference | 0 | 0 | $0.00 | $0.00 |
| 2026-07 | C_market_logistic | 109 | 19 | $15.58 | $-135.63 |
| 2026-08 | C_market_logistic | 157 | 17 | $44.79 | $-140.25 |
| 2026-09 | C_market_logistic | 50 | 16 | $5.41 | $12.62 |
| 2026-10 | C_market_logistic | 13 | 3 | $0.40 | $-31.27 |
| 2026-07 | D_market_plus_BTC | 2,019 | 25 | $295.30 | $-552.57 |
| 2026-08 | D_market_plus_BTC | 1,942 | 27 | $355.90 | $114.17 |
| 2026-09 | D_market_plus_BTC | 2,062 | 30 | $231.19 | $108.53 |
| 2026-10 | D_market_plus_BTC | 592 | 6 | $63.38 | $-305.63 |
| 2026-07 | E_probability_mixture | 1,824 | 25 | $401.78 | $-494.40 |
| 2026-08 | E_probability_mixture | 1,788 | 27 | $582.41 | $108.68 |
| 2026-09 | E_probability_mixture | 1,755 | 30 | $238.31 | $30.06 |
| 2026-10 | E_probability_mixture | 504 | 6 | $59.63 | $-201.64 |
| 2026-07 | F_regularized_logit_correction | 0 | 0 | $0.00 | $0.00 |
| 2026-08 | F_regularized_logit_correction | 0 | 0 | $0.00 | $0.00 |
| 2026-09 | F_regularized_logit_correction | 0 | 0 | $0.00 | $0.00 |
| 2026-10 | F_regularized_logit_correction | 0 | 0 | $0.00 | $0.00 |

### Wykonalność zamrożonej polityki przy 5% wolnej gotówki

Kapitał początkowy $100; stake 5% wolnej gotówki, minimum obowiązuje, bez podbijania stawki. Fille 2; obrót $10.24; fee $0.36; PnL $-0.60; saldo końcowe $99.40; max DD 5.17%; audyt `passed`.

## Odpowiedzi

1. W rozwoju rynek B ma log-loss 0.69299 wobec 0.69231 dla BTC A; Brier to 0.24992 wobec 0.24958. B ma niższy LL i ECE niż C (+0.00080 dla C−B; ECE C=0.0271) oraz niższe ECE niż D (0.0238), ale D ma nieco niższy LL (-0.00012 dla D−B). A ma niższy LL niż B. E ma punktowo log-loss 0.69242 i Brier 0.24963; E-B ΔLL -0.00057 [-0.00124, +0.00013] oraz ΔBrier -0.00028 [-0.00062, +0.00006]. Przedziały obejmują zero, a E ECE 0.0181 jest gorsze od B. Zakotwiczenie rynkowe poprawia ECE względem pozostałych prognoz, ale nie daje stabilnego zysku w LL; efekt BTC również nie jest stabilny.
2. Nie ma wiarygodnego dowodu, że BTC daje poprawę ponad rynek: E-B CI obejmuje zero; E-D ΔLL -0.00045 [-0.00106, +0.00018] też obejmuje zero. F-B ma ΔLL +0.00023 [-0.00011, +0.00060], a końcowy wybór F to F_no_correction (dokładnie B, bez korekty). Końcowy wybór E to E_w_1 (dokładnie A, sama prognoza BTC). Brak podstaw do wyboru aktywnej korekty E/F; no-trade pozostaje kontrolą dla korekty ponad rynek.
3. Na koszyku wykonań D korekta E poprawia punktowo log-loss 0.69195 vs 0.69256, a lukę EV-PnL zmienia z $293.55 do $-16.49; oba 95% CI luki obejmują zero. Na stałym koszyku A E jest punktowo gorsze od D (0.69112 vs 0.69090). Poprawa nie powtarza się na obu zestawach wykonań.
4. E przy hurdle 1% ma development PnL $102.16 po fee na 462 fillach, ale 95% blokowe CI [$-53.48, $286.62] obejmuje zero. Koszty są wliczone z rzeczywistego T-59. Wybrana development polityka to A_BTC_calibrated_ev_1.0000%; na stałych $5 ma $264.52 (CI [$37.74, $537.86]), ale jest to BTC-only A, a przedział nie koryguje selekcji spośród modeli i progów. Portfel 5% wolnej gotówki na historycznym holdoucie wykonał 2 zlecenia i stracił $0.60; wynik ten nie potwierdza poprawki E/F.
5. Wybrana A polityka w development miała 1,035 filli w 43 dniach; PnL foldów: maj $87.53, czerwiec $176.99. Trzy największe dni to 3.0% bezwzględnego wyniku. Zatem development wynik A rozkłada się na wiele dni, ale holdout feasibility ograniczyło minimum do 2 filli; po tym nie podbijano stawki. Holdout B ma LL 0.69158, a E 0.69245; to historyczna diagnostyka, bo okres był wcześniej oglądany.

Skrócone tabele CSV: [`btc_market_probability_nested_20261012/`](btc_market_probability_nested_20261012/). Pełne predykcje i ledgery: `data/analysis/polymarket/BTC/btc_market_probability_nested_20261012`. Konfigurację i freeze development zapisano przed metrykami holdout. Model BTC nie był trenowany, replay księgi nie był powtarzany, a konfiguracja live nie była zmieniana.
