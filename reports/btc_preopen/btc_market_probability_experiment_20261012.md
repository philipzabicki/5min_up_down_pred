# BTC i rynek: porównanie prognoz 5m — 12 października 2026

> Holdout lipiec–październik był wcześniej oglądany. To zamknięta diagnostyka historyczna, nie nowe niezależne potwierdzenie.
> Skorygowany raport błędu T−59: [diagnostyka sygnału](btc_preopen_signal_diagnostic_correction_20261012.md).

## Werdykt

Na wspólnych 24,736 rynkach holdout najniższy log loss ma B_market_reference (0.69158); rynek B: 0.69158, BTC A: 0.69245, C: 0.69171, D: 0.69176. D−C log-loss=+0.00005 (95% CI -0.00062…+0.00060); D−A=-0.00069 (95% CI -0.00142…-0.00013). Zamrożony wybór portfela na development: C_market_logistic / 0.00%.

## Zamrożone zasady i pokrycie

T−60 wspólne kwotowania: development 10,795/47,483 rynków (22.7%); holdout 24,736/28,221 (87.7%). Braki cen pominięto bez imputacji.
Poprawne obustronne kwotowanie wymaga `quote_valid`, pokrycia godziny źródłowej, obu bid/ask w zakresie i bez crossing, braku przyszłych zdarzeń oraz braku udokumentowanej luki po inicjalizacji booka. Ask age nie jest filtrem. Udział źródeł i proxy: ag6_v2/full_native_snapshot: 994; ag6_v2/proxy_or_complemented: 36; pmxt_v2/full_native_snapshot: 9,387; v3/full_native_snapshot: 14,319.
T−60 kwotowania na populacji z potwierdzonym wynikiem i przyczynową prognozą:

| Okres | UP dostępne | DOWN dostępne | Obie strony dostępne | Próba wspólna | Mediana UP bid/ask | UP spread | Mediana DOWN bid/ask | DOWN spread | Proxy/complemented |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| development | 10,864 | 11,261 | 10,840 | 10,795 | 0.500 / 0.510 | 0.010 | 0.490 / 0.500 | 0.010 | 0 |
| holdout | 24,741 | 24,742 | 24,740 | 24,736 | 0.500 / 0.510 | 0.010 | 0.490 / 0.500 | 0.010 | 36 |

Referencja: `m_up=(bid_up+ask_up)/2`, `m_down=(bid_down+ask_down)/2`, `p_market=m_up/(m_up+m_down)`. Jest prognozą, nie ceną wykonania.
C i D wybrano na dwóch chronologicznych foldach rozwoju, z oficjalnym wynikiem i etykietami dostępnymi przed początkiem walidacji. Fold 15–30 kwietnia nie mógł służyć do doboru C/D, ponieważ nie było wcześniejszej próby księgi. Rozmiar próby nie uzasadnia bardziej złożonego modelu.

## Dobór regularizacji i współczynniki

| Model | C wybrane | λ=1/C | CV log loss | CV rynki |
|---|---:|---:|---:|---:|
| C | 10.0000 | 0.1000 | 0.69356 | 8,676 |
| D | 10.0000 | 0.1000 | 0.69286 | 8,676 |

Współczynniki w skali logitów; modele ponownie estymowano na tym samym prefiksie przed każdym foldem:

| Model | Fold | n train | n valid | Intercept | β logit rynku | β logit BTC |
|---|---|---:|---:|---:|---:|---:|
| C | dev_2026q2_b | 2,119 | 5,253 | 0.0669 | 0.9770 | 0.0000 |
| D | dev_2026q2_b | 2,119 | 5,253 | 0.0645 | 0.1187 | 1.1798 |
| C | dev_2026q2_c | 7,371 | 3,423 | 0.0177 | 0.7238 | 0.0000 |
| D | dev_2026q2_c | 7,371 | 3,423 | 0.0078 | 0.5158 | 0.9727 |

Rozrzut współczynników między dwoma foldami:

| Model | Współczynnik | Średnia | SD | Min | Max | Zgodność znaku |
|---|---|---:|---:|---:|---:|---:|
| C | intercept | 0.0423 | 0.0246 | 0.0177 | 0.0669 | 2/2 |
| C | logit_market | 0.8504 | 0.1266 | 0.7238 | 0.9770 | 2/2 |
| D | intercept | 0.0362 | 0.0283 | 0.0078 | 0.0645 | 2/2 |
| D | logit_market | 0.3172 | 0.1985 | 0.1187 | 0.5158 | 2/2 |
| D | logit_btc_calibrated | 1.0762 | 0.1036 | 0.9727 | 1.1798 | 2/2 |

## Jakość prawdopodobieństw na tych samych rynkach

A oznacza dotychczasowe skalibrowane P(BTC); dodatkowo raport surowego P(BTC). C/D w development to wyłącznie chronologiczne prognozy OOF. Holdout korzysta z modeli dopasowanych po zamrożeniu ustawień na rozwojowym OOF.

| Okres | Prognoza | n | UP | Log loss | Brier | AUC | Slope / intercept | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| development_cv | A_BTC_calibrated | 8,676 | 49.20% | 0.69231 | 0.24958 | 0.5251 | 1.106 / -0.049 | 0.0224 |
| holdout | A_BTC_calibrated | 24,736 | 49.83% | 0.69245 | 0.24965 | 0.5227 | 0.800 / -0.008 | 0.0091 |
| development_cv | B_market_reference | 8,676 | 49.20% | 0.69299 | 0.24992 | 0.5157 | 0.747 / -0.045 | 0.0135 |
| holdout | B_market_reference | 24,736 | 49.83% | 0.69158 | 0.24923 | 0.5202 | 1.053 / -0.012 | 0.0084 |
| development_cv | C_market_logistic | 8,676 | 49.20% | 0.69356 | 0.25020 | 0.5230 | 0.953 / -0.093 | 0.0248 |
| holdout | C_market_logistic | 24,736 | 49.83% | 0.69171 | 0.24929 | 0.5202 | 1.397 / 0.020 | 0.0052 |
| development_cv | D_market_plus_BTC | 8,676 | 49.20% | 0.69286 | 0.24986 | 0.5288 | 1.009 / -0.095 | 0.0248 |
| holdout | D_market_plus_BTC | 24,736 | 49.83% | 0.69176 | 0.24931 | 0.5289 | 0.896 / 0.019 | 0.0108 |
| development_cv | A_BTC_raw | 8,676 | 49.20% | 0.69231 | 0.24958 | 0.5250 | 1.309 / -0.045 | 0.0207 |
| holdout | A_BTC_raw | 24,736 | 49.83% | 0.69241 | 0.24963 | 0.5227 | 0.926 / -0.012 | 0.0078 |

Sparowane różnice holdout `D−C` i `D−A`; dla log loss/Brier wartość ujemna sprzyja D, dla AUC dodatnia. Przedziały pochodzą z tych samych losowań trzydniowych bloków UTC:

| Porównanie | Δ log loss [95% CI] | Δ Brier [95% CI] | Δ AUC [95% CI] |
|---|---:|---:|---:|
| D-C | 0.00005 [-0.00062, 0.00060] | 0.00002 [-0.00031, 0.00029] | 0.0087 [0.0044, 0.0144] |
| D-A | -0.00069 [-0.00142, -0.00013] | -0.00034 [-0.00071, -0.00006] | 0.0062 [0.0021, 0.0117] |

### Wyniki miesięczne jakości prognoz

| Miesiąc UTC | Prognoza | n | Log loss | Brier | AUC |
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

## Niezależne zlecenia $5: T−60 decyzja, T−59 wykonanie

Dla każdej prognozy użyto tych samych progów EV. Strona i żądane $5 pochodzą z T−60; wykonanie ponownie wyceniono po ask/VWAP T−59 z zaakceptowaną opłatą, minimum, FAK i znaną głębokością albo jawnie oznaczonym proxy. Sygnały nie współdzielą gotówki.

| Okres | Prognoza | EV min. | Sygnały | Fille | Obrót | Fee eq. | Oczekiwany EV T−59 | Zrealizowany PnL netto | EV−PnL / fill | Netto / obrót | Full book / proxy / partial |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| development_cv | A_BTC_calibrated | 0.00% | 1,621 | 1,606 | $8,030.00 | $286.48 | $254.21 | $200.77 | $0.033 | 2.50% | 1,606 / 0 / 0 |
| development_cv | A_BTC_calibrated | 0.25% | 1,456 | 1,443 | $7,215.00 | $257.87 | $253.50 | $213.01 | $0.028 | 2.95% | 1,443 / 0 / 0 |
| development_cv | A_BTC_calibrated | 0.50% | 1,304 | 1,292 | $6,460.00 | $231.48 | $250.70 | $246.83 | $0.003 | 3.82% | 1,292 / 0 / 0 |
| development_cv | A_BTC_calibrated | 1.00% | 1,044 | 1,035 | $5,175.00 | $186.51 | $241.51 | $264.52 | $-0.022 | 5.11% | 1,035 / 0 / 0 |
| development_cv | A_BTC_calibrated | 2.00% | 649 | 640 | $3,200.00 | $117.08 | $212.51 | $161.63 | $0.080 | 5.05% | 640 / 0 / 0 |
| development_cv | B_market_reference | 0.00% | 0 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | n/a | n/a% | 0 / 0 / 0 |
| development_cv | B_market_reference | 0.25% | 0 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | n/a | n/a% | 0 / 0 / 0 |
| development_cv | B_market_reference | 0.50% | 0 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | n/a | n/a% | 0 / 0 / 0 |
| development_cv | B_market_reference | 1.00% | 0 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | n/a | n/a% | 0 / 0 / 0 |
| development_cv | B_market_reference | 2.00% | 0 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | n/a | n/a% | 0 / 0 / 0 |
| development_cv | C_market_logistic | 0.00% | 10 | 10 | $50.00 | $2.14 | $1.45 | $38.47 | $-3.702 | 76.94% | 10 / 0 / 0 |
| development_cv | C_market_logistic | 0.25% | 10 | 10 | $50.00 | $2.14 | $1.45 | $38.47 | $-3.702 | 76.94% | 10 / 0 / 0 |
| development_cv | C_market_logistic | 0.50% | 10 | 10 | $50.00 | $2.14 | $1.45 | $38.47 | $-3.702 | 76.94% | 10 / 0 / 0 |
| development_cv | C_market_logistic | 1.00% | 8 | 8 | $40.00 | $1.74 | $1.39 | $25.61 | $-3.028 | 64.04% | 8 / 0 / 0 |
| development_cv | C_market_logistic | 2.00% | 6 | 6 | $30.00 | $1.32 | $1.25 | $23.21 | $-3.660 | 77.38% | 6 / 0 / 0 |
| development_cv | D_market_plus_BTC | 0.00% | 2,173 | 2,146 | $10,730.00 | $373.92 | $312.63 | $-22.06 | $0.156 | -0.21% | 2,146 / 0 / 0 |
| development_cv | D_market_plus_BTC | 0.25% | 1,993 | 1,968 | $9,840.00 | $343.19 | $311.64 | $20.69 | $0.148 | 0.21% | 1,968 / 0 / 0 |
| development_cv | D_market_plus_BTC | 0.50% | 1,838 | 1,816 | $9,080.00 | $316.99 | $309.05 | $26.20 | $0.156 | 0.29% | 1,816 / 0 / 0 |
| development_cv | D_market_plus_BTC | 1.00% | 1,514 | 1,495 | $7,475.00 | $261.78 | $297.62 | $170.42 | $0.085 | 2.28% | 1,495 / 0 / 0 |
| development_cv | D_market_plus_BTC | 2.00% | 1,022 | 1,009 | $5,045.00 | $177.74 | $263.09 | $57.96 | $0.203 | 1.15% | 1,009 / 0 / 0 |
| holdout | A_BTC_calibrated | 0.00% | 6,248 | 5,871 | $29,355.00 | $1,047.12 | $1,282.12 | $-557.30 | $0.313 | -1.90% | 5,871 / 0 / 0 |
| holdout | A_BTC_calibrated | 0.25% | 5,705 | 5,358 | $26,790.00 | $958.31 | $1,280.36 | $-578.32 | $0.347 | -2.16% | 5,358 / 0 / 0 |
| holdout | A_BTC_calibrated | 0.50% | 5,176 | 4,862 | $24,310.00 | $872.20 | $1,272.16 | $-686.83 | $0.403 | -2.83% | 4,862 / 0 / 0 |
| holdout | A_BTC_calibrated | 1.00% | 4,272 | 4,018 | $20,090.00 | $725.31 | $1,242.33 | $-732.16 | $0.491 | -3.64% | 4,018 / 0 / 0 |
| holdout | A_BTC_calibrated | 2.00% | 2,872 | 2,707 | $13,535.00 | $497.51 | $1,149.97 | $-449.26 | $0.591 | -3.32% | 2,707 / 0 / 0 |
| holdout | B_market_reference | 0.00% | 0 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | n/a | n/a% | 0 / 0 / 0 |
| holdout | B_market_reference | 0.25% | 0 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | n/a | n/a% | 0 / 0 / 0 |
| holdout | B_market_reference | 0.50% | 0 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | n/a | n/a% | 0 / 0 / 0 |
| holdout | B_market_reference | 1.00% | 0 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | n/a | n/a% | 0 / 0 / 0 |
| holdout | B_market_reference | 2.00% | 0 | 0 | $0.00 | $0.00 | $0.00 | $0.00 | n/a | n/a% | 0 / 0 / 0 |
| holdout | C_market_logistic | 0.00% | 336 | 329 | $1,645.00 | $74.07 | $66.19 | $-294.52 | $1.096 | -17.90% | 329 / 0 / 0 |
| holdout | C_market_logistic | 0.25% | 322 | 316 | $1,580.00 | $71.31 | $66.14 | $-303.69 | $1.170 | -19.22% | 316 / 0 / 0 |
| holdout | C_market_logistic | 0.50% | 290 | 285 | $1,425.00 | $65.09 | $65.82 | $-330.58 | $1.391 | -23.20% | 285 / 0 / 0 |
| holdout | C_market_logistic | 1.00% | 247 | 244 | $1,220.00 | $56.48 | $64.61 | $-271.86 | $1.379 | -22.28% | 244 / 0 / 0 |
| holdout | C_market_logistic | 2.00% | 209 | 206 | $1,030.00 | $48.35 | $61.76 | $-236.71 | $1.449 | -22.98% | 206 / 0 / 0 |
| holdout | D_market_plus_BTC | 0.00% | 6,937 | 6,615 | $33,075.00 | $1,156.66 | $945.76 | $-635.50 | $0.239 | -1.92% | 6,615 / 0 / 0 |
| holdout | D_market_plus_BTC | 0.25% | 6,404 | 6,106 | $30,530.00 | $1,068.94 | $942.28 | $-539.24 | $0.243 | -1.77% | 6,106 / 0 / 0 |
| holdout | D_market_plus_BTC | 0.50% | 5,890 | 5,618 | $28,090.00 | $985.22 | $933.85 | $-452.79 | $0.247 | -1.61% | 5,618 / 0 / 0 |
| holdout | D_market_plus_BTC | 1.00% | 4,890 | 4,677 | $23,385.00 | $823.34 | $898.72 | $-571.29 | $0.314 | -2.44% | 4,677 / 0 / 0 |
| holdout | D_market_plus_BTC | 2.00% | 3,221 | 3,083 | $15,415.00 | $548.96 | $786.74 | $-394.38 | $0.383 | -2.56% | 3,083 / 0 / 0 |

### Holdout miesięcznie przy progu EV 0%

| Miesiąc UTC | Prognoza | Fille | Oczekiwany EV T−59 | PnL netto | EV−PnL / fill |
|---|---|---:|---:|---:|---:|
| 2026-07 | A_BTC_calibrated | 1,824 | $401.78 | $-494.40 | $0.491 |
| 2026-08 | A_BTC_calibrated | 1,788 | $582.41 | $108.68 | $0.265 |
| 2026-09 | A_BTC_calibrated | 1,755 | $238.31 | $30.06 | $0.119 |
| 2026-10 | A_BTC_calibrated | 504 | $59.63 | $-201.64 | $0.518 |
| 2026-07 | B_market_reference | 0 | $0.00 | $0.00 | n/a |
| 2026-08 | B_market_reference | 0 | $0.00 | $0.00 | n/a |
| 2026-09 | B_market_reference | 0 | $0.00 | $0.00 | n/a |
| 2026-10 | B_market_reference | 0 | $0.00 | $0.00 | n/a |
| 2026-07 | C_market_logistic | 109 | $15.58 | $-135.63 | $1.387 |
| 2026-08 | C_market_logistic | 157 | $44.79 | $-140.25 | $1.179 |
| 2026-09 | C_market_logistic | 50 | $5.41 | $12.62 | $-0.144 |
| 2026-10 | C_market_logistic | 13 | $0.40 | $-31.27 | $2.436 |
| 2026-07 | D_market_plus_BTC | 2,019 | $295.30 | $-552.57 | $0.420 |
| 2026-08 | D_market_plus_BTC | 1,942 | $355.90 | $114.17 | $0.124 |
| 2026-09 | D_market_plus_BTC | 2,062 | $231.19 | $108.53 | $0.059 |
| 2026-10 | D_market_plus_BTC | 592 | $63.38 | $-305.63 | $0.623 |

## Zamrożona symulacja portfela 5% wolnej gotówki

Na development OOF od 2026-05-01T00:00:00+00:00 do 2026-07-01T00:00:00+00:00 wybrano `C_market_logistic_ev_0.0000%` przy hurdle 0.0; kontrola no-trade kończy z $100.00. Wybór i parametry zapisano przed otwarciem wyników holdout w `reports/btc_preopen/btc_market_probability_frozen_selection_20261012.json`.

| Prognoza | EV min. | Końcowa gotówka development | PnL | Transakcje | Max DD |
|---|---:|---:|---:|---:|---:|
| A_BTC_calibrated | 0.00% | $94.82 | $-5.18 | 1 | 5.18% |
| A_BTC_calibrated | 0.25% | $94.82 | $-5.18 | 1 | 5.18% |
| A_BTC_calibrated | 0.50% | $94.82 | $-5.18 | 1 | 5.18% |
| A_BTC_calibrated | 1.00% | $94.82 | $-5.18 | 1 | 5.18% |
| A_BTC_calibrated | 2.00% | $99.39 | $-0.61 | 2 | 5.18% |
| B_market_reference | 0.00% | $100.00 | $0.00 | 0 | 0.00% |
| B_market_reference | 0.25% | $100.00 | $0.00 | 0 | 0.00% |
| B_market_reference | 0.50% | $100.00 | $0.00 | 0 | 0.00% |
| B_market_reference | 1.00% | $100.00 | $0.00 | 0 | 0.00% |
| B_market_reference | 2.00% | $100.00 | $0.00 | 0 | 0.00% |
| C_market_logistic | 0.00% | $143.34 | $43.34 | 10 | 5.44% |
| C_market_logistic | 0.25% | $143.34 | $43.34 | 10 | 5.44% |
| C_market_logistic | 0.50% | $143.34 | $43.34 | 10 | 5.44% |
| C_market_logistic | 1.00% | $94.79 | $-5.21 | 1 | 5.21% |
| C_market_logistic | 2.00% | $124.06 | $24.06 | 6 | 5.44% |
| D_market_plus_BTC | 0.00% | $95.88 | $-4.12 | 12 | 20.43% |
| D_market_plus_BTC | 0.25% | $95.88 | $-4.12 | 12 | 20.43% |
| D_market_plus_BTC | 0.50% | $96.47 | $-3.53 | 10 | 19.80% |
| D_market_plus_BTC | 1.00% | $97.23 | $-2.77 | 8 | 15.42% |
| D_market_plus_BTC | 2.00% | $98.78 | $-1.22 | 4 | 10.09% |

### Holdout portfel selected policy — osobny kapitał początkowy

| Kapitał początkowy | Końcowa gotówka | PnL netto | Transakcje | Obrót | Fee | Max DD | Audyt |
|---:|---:|---:|---:|---:|---:|---:|---|
| $100 | $95.67 | $-4.33 | 8 | $42.16 | $1.81 | 16.75% | passed |
| $500 | $99.04 | $-400.96 | 91 | $1,283.85 | $55.68 | 83.98% | passed |
| $1,000 | $95.30 | $-904.70 | 169 | $3,267.89 | $143.49 | 92.28% | passed |

## Rozstrzygnięcie

1. T−59 poprawia wcześniejszą diagnostykę: poprzednia wycena wykonania używała T−60. Po korekcie raportowane są rzeczywiste snapshoty T−59; osobne portfele były poprawne i pozostają bez przeliczenia.
2. Referencja B ma log loss 0.69158 i Brier 0.24923, lepsze od BTC A (0.69245 i 0.24965); BTC A ma wyższe AUC (0.5227 vs 0.5202). B nie daje sygnałów z dodatnim EV ask po fee.
3. Dodanie BTC do rynku podnosi AUC D vs C o 0.0087 (95% CI 0.0044 do 0.0144), lecz nie poprawia log loss (+0.00005) ani Brier (+0.00002); oba przedziały obejmują zero. Dodatni współczynnik BTC sam nie wystarcza jako dowód poprawy prawdopodobieństw.
4. Przy EV 0% luka EV minus PnL na fill spada dla D do $0.239 z $0.313 dla A (o 23.7%); suma luki to $1,581 dla D wobec $1,839 dla A (o 14.1%). Nie poprawia to wyniku po fee: PnL netto D to -$635, A -$557 (ROI -1.92% i -1.90%). Referencja B nie wygenerowała sygnałów z dodatnim EV ask po fee.
5. Wynik nie utrzymuje się w każdym miesiącu: D ma najlepszy log loss w sierpniu i wrześniu, ale nie w lipcu ani październiku; PnL D przy progu 0% wynosi odpowiednio -$553, +$114, +$109 i -$306. Wybór portfela z OOF development pozostaje C_market_logistic / 0.00%; holdout był wcześniej oglądany.

Odtworzenie: `python run_btc_preopen_signal_diagnostic.py`, następnie `python run_btc_market_probability_experiment.py`. Manifest i szczegółowe tabele znajdują się w `data/analysis/polymarket/BTC/btc_market_probability_experiment_20261012`. Model BTC nie był trenowany ani modyfikowany; replay księgi i live nie uruchamiano.
