# Badanie DRK, RCK i OS — 7 października 2026

Badanie używa zamrożonego `candidate_platt`, wejścia T−59 s, kapitału $100, braku kredytu i limitu zakupu $20. To retrospektywna walidacja rozwojowa na okresie wcześniej używanym do selekcji, a nie nowy niezależny holdout.

## Rekonstrukcja baseline’u

Odtworzony dokładny fill $5: **$+475.840156**. Baseline starej reguły na zapisanym best level: **$+477.917696** (różnica $+2.077540, wynik modelu wykonania). Obie wartości są dla pełnego okresu rozwojowego. Wspólny zewnętrzny okres obejmuje 2673 rynków w trzech zamrożonych blokach.

## Wspólna tabela polityk na zewnętrznych blokach

| Polityka | Transakcje | PnL | Śr. dzienny log wzrostu | Max DD kosztowy | Czas pod wodą (dni) | Max ekspozycja | Min. gotówka | Opłaty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_trade | 0 | $0.00 | 0.000% | 0.00% | 0.00 | $0.00 | $100.00 | $0.0000 |
| legacy_best_level_fixed_5 | 2244 | $461.15 | 8.213% | 63.82% | 5.19 | $15.00 | $27.99 | $0.0000 |
| legacy_exact_fixed_5 | 2285 | $461.42 | 8.216% | 75.37% | 5.29 | $15.00 | $15.87 | $0.0000 |
| fractional_kelly_half_10pct | 529 | $209.65 | 5.382% | 32.62% | 6.99 | $23.33 | $83.21 | $0.0000 |
| point_kelly_btc_only | 1779 | $453.87 | 8.151% | 89.52% | 12.44 | $40.00 | $17.22 | $0.0000 |
| drk_btc_only | 265 | $-14.39 | -0.740% | 62.20% | 19.61 | $17.40 | $35.25 | $0.0000 |
| point_kelly_btc_market | 1824 | $312.08 | 6.743% | 86.14% | 13.12 | $40.00 | $24.26 | $0.0000 |
| drk_btc_market | 285 | $-49.07 | -3.213% | 70.24% | 19.61 | $15.41 | $27.16 | $0.0000 |
| portfolio_kelly_independent | 1827 | $319.19 | 6.825% | 86.36% | 13.12 | $40.00 | $23.81 | $0.0000 |
| rck_alpha_0.5_correlated | 303 | $33.52 | 1.377% | 39.77% | 12.90 | $24.27 | $61.17 | $0.0000 |
| rck_alpha_0.5_independent | 281 | $29.88 | 1.245% | 42.79% | 12.91 | $24.26 | $57.60 | $0.0000 |
| rck_alpha_0.7_correlated | 26 | $-4.18 | -0.203% | 21.86% | 19.61 | $12.87 | $71.02 | $0.0000 |
| rck_alpha_0.7_independent | 26 | $-4.22 | -0.205% | 21.86% | 19.61 | $12.87 | $70.97 | $0.0000 |
| rck_alpha_0.8_correlated | 4 | $-5.95 | -0.292% | 10.86% | 19.61 | $9.12 | $80.02 | $0.0000 |
| rck_alpha_0.8_independent | 4 | $-5.95 | -0.292% | 10.86% | 19.61 | $9.12 | $80.02 | $0.0000 |

PnL, drawdown i czas pod wodą pochodzą z tego samego chronologicznego ledgeru. Ekspozycja to koszt brutto nierozliczonych pozycji; nie jest wyceną likwidacyjną. Wypełnienie zakupu przyjęto na zapisanym best ask i ilości, a historyczne opłaty liczy wspólny ledger.

## DRK

Wariant DRK z ceną rynku nie poprawił PnL względem odpowiadającego mu punktowego Kelly; sama odporność na niepewność nie wykazała stabilnej korzyści.
Punktowa kalibracja BTC+midpoint rynku: PnL $312.08; DRK na tym samym modelu z przedziałem bootstrapowym: $-49.07. Porównanie BTC-only i BTC+market jest w tabeli oraz `advanced_policy_study_20261007.json`; C wybierano wyłącznie na poprzedzającym bloku wewnętrznym, a bootstrap używał 500 replik 3-dniowych bloków.

## RCK

Nie ma stabilnej poprawy RCK nad fractional Kelly w dostępnych blokach; profil ryzyka należy czytać jako lokalne ograniczenie scenariuszowe, nie gwarancję drawdownu.
RCK używa wspólnych scenariuszy AR(1) copula oraz wariantu niezależnego, a także trzech intensywności α=0.5/0.7/0.8 przy β=0.1. Mianownikiem jest cost-basis equity zamrożone przed decyzją; to jawne przybliżenie wartości majątku, a nie wycena mark-to-market. Środki z otwartych pozycji wchodzą do terminalnych wypłat scenariuszowych raz, lecz nie finansują bieżącego zakupu. Gdy scenariuszowy portfel już narusza lokalny limit, dodatkowy zakup jest blokowany.

## 1. Pełne polityki na wspólnej populacji

Tabela wyżej pokazuje ciągły portfel na trzech outer blokach, z oddzielnym przebiegiem dla każdego wariantu i początkowym kapitałem $100.

## 2. Selekcja przy wspólnej stawce $5

| Reguła wyboru | Rynki | Wybrane | Wykonalne $5 | UP / DOWN | Win rate | PnL | Śr. PnL / trade |
|---|---:|---:|---:|---:|---:|---:|---:|
| candidate_platt | 2673 | 955 | 955 | 197 / 758 | 53.51% | $357.44 | $0.3743 |
| point_kelly_btc_only | 2673 | 1356 | 1356 | 1089 / 267 | 52.95% | $317.74 | $0.2343 |
| drk_btc_only | 2673 | 240 | 240 | 230 / 10 | 52.50% | $56.04 | $0.2335 |
| point_kelly_btc_market | 2673 | 1330 | 1330 | 1148 / 182 | 53.68% | $382.76 | $0.2878 |
| drk_btc_market | 2673 | 266 | 266 | 266 / 0 | 50.75% | $-0.14 | $-0.0005 |

Każdy rynek jest oceniany osobno bez reinwestowania i bez wspólnego ograniczenia gotówkowego; wykonanie i minima pozostają wspólne. To izoluje wybór strony od sizingu.

## 3. Sizing na zamrożonym market/side/time

| Sizing | Zamrożone wejścia | Wykonalne | Wykonalność | Śr. żądana | Śr. przyjęta | PnL na wykonalnych | Ograniczone top-level |
|---|---:|---:|---:|---:|---:|---:|---:|
| candidate_fractional_kelly | 2285 | 290 | 12.69% | $1.28 | $3.26 | $112.38 | 15 |
| point_kelly_btc_only | 2285 | 949 | 41.53% | $2.53 | $5.45 | $292.44 | 55 |
| drk_btc_only | 2285 | 221 | 9.67% | $0.70 | $3.78 | $16.09 | 7 |
| point_kelly_btc_market | 2285 | 792 | 34.66% | $2.23 | $5.60 | $212.95 | 54 |
| drk_btc_market | 2285 | 202 | 8.84% | $0.60 | $4.22 | $-19.75 | 12 |
| rck_alpha_0.5 | 2285 | 405 | 17.72% | $0.61 | $3.43 | $18.01 | 0 |
| rck_alpha_0.7 | 2285 | 54 | 2.36% | $0.07 | $3.12 | $19.65 | 0 |
| rck_alpha_0.8 | 2285 | 4 | 0.18% | $0.01 | $3.49 | $-8.68 | 0 |

Lista market/side/time pochodzi ze starej reguły wyboru strony na dokładnym fillu $5. Tabela pokazuje PnL tylko dla wykonalnych zleceń, a osobno liczbę odrzuceń i ograniczeń ilości.

## 4. Wyjścia na identycznych zakupach i portfel

| Reguła wyjścia | Wynik |
|---|---|
| Hold-to-resolution | dostępna kontrola; zakupy i rozliczenia z ledgeru |
| Prosta reguła sprzedaży | oceniona w osobnym raporcie `optimal_stopping_20261007.md` |
| Regresyjne OS | oceniono w osobnym raporcie `optimal_stopping_20261007.md` |

Audyt znalazł 4454/4454 rynków z 300 próbkami jedn-sekundowymi i bez luk. Dane Kacho zawierają bid, ask i ilość na najlepszym poziomie; brak ilości oznacza brak wykonalnej sprzedaży na tym poziomie. `resolved_at_utc` jest jawnym przybliżeniem dostępności wyniku, a nie zarejestrowanym czasem odbioru etykiety. Per-market coverage: `reports/btc_preopen/trajectory_coverage_per_market_20261007.csv`. Dodatkowe pobranie PMXT nie było potrzebne; szczegóły OS są w osobnym raporcie kontynuacji.

## Kalibracja i zależność scenariuszy

| Blok | Model | Wybrane C | Szerokość przedziału p (średnia) | ρ AR(1) po shrinkage |
|---:|---|---:|---:|---:|
| 1 | btc_only | 1 | 0.0425 | 0.0219 |
| 1 | btc_market | 1 | 0.0404 | 0.0219 |
| 2 | btc_only | 10 | 0.0373 | 0.0125 |
| 2 | btc_market | 10 | 0.0407 | 0.0125 |
| 3 | btc_only | 10 | 0.0334 | 0.0082 |
| 3 | btc_market | 10 | 0.0338 | 0.0082 |

C wybierano wyłącznie na poprzedzającym bloku wewnętrznym. Przedziały p są kwantylami 5–95% z 500 bootstrapów 3-dniowych bloków, nie gwarantowanymi przedziałami prawdziwego prawdopodobieństwa.

## Wyniki między blokami

| Polityka | Blok | Transakcje | PnL transakcji z bloku | Equity po decyzjach | Zmiany przez RCK |
|---|---:|---:|---:|---:|---:|
| no_trade | 1 | 0 | $0.00 | $100.00 | 0 |
| no_trade | 2 | 0 | $0.00 | $100.00 | 0 |
| no_trade | 3 | 0 | $0.00 | $100.00 | 0 |
| legacy_best_level_fixed_5 | 1 | 734 | $156.91 | $256.91 | 0 |
| legacy_best_level_fixed_5 | 2 | 755 | $-39.45 | $212.46 | 0 |
| legacy_best_level_fixed_5 | 3 | 755 | $343.69 | $566.15 | 0 |
| legacy_exact_fixed_5 | 1 | 750 | $161.69 | $261.69 | 0 |
| legacy_exact_fixed_5 | 2 | 767 | $-60.80 | $195.89 | 0 |
| legacy_exact_fixed_5 | 3 | 768 | $360.53 | $566.42 | 0 |
| fractional_kelly_half_10pct | 1 | 125 | $18.47 | $118.47 | 0 |
| fractional_kelly_half_10pct | 2 | 101 | $12.23 | $130.70 | 0 |
| fractional_kelly_half_10pct | 3 | 303 | $178.95 | $309.65 | 0 |
| point_kelly_btc_only | 1 | 648 | $-22.08 | $75.47 | 0 |
| point_kelly_btc_only | 2 | 423 | $35.64 | $110.96 | 0 |
| point_kelly_btc_only | 3 | 708 | $440.31 | $573.87 | 0 |
| drk_btc_only | 1 | 57 | $-56.90 | $43.10 | 0 |
| drk_btc_only | 2 | 49 | $1.90 | $45.00 | 0 |
| drk_btc_only | 3 | 159 | $40.61 | $88.16 | 0 |
| point_kelly_btc_market | 1 | 649 | $-27.73 | $69.82 | 0 |
| point_kelly_btc_market | 2 | 505 | $98.26 | $167.91 | 0 |
| point_kelly_btc_market | 3 | 670 | $241.55 | $430.76 | 0 |
| drk_btc_market | 1 | 55 | $-46.84 | $53.16 | 0 |
| drk_btc_market | 2 | 48 | $-7.50 | $45.66 | 0 |
| drk_btc_market | 3 | 182 | $5.27 | $50.93 | 0 |
| portfolio_kelly_independent | 1 | 647 | $-28.02 | $69.53 | 0 |
| portfolio_kelly_independent | 2 | 507 | $107.53 | $176.75 | 0 |
| portfolio_kelly_independent | 3 | 673 | $239.68 | $438.19 | 0 |
| rck_alpha_0.5_correlated | 1 | 40 | $-27.72 | $72.28 | 605 |
| rck_alpha_0.5_correlated | 2 | 68 | $9.34 | $81.62 | 518 |
| rck_alpha_0.5_correlated | 3 | 195 | $51.90 | $136.25 | 589 |
| rck_alpha_0.5_independent | 1 | 42 | $-32.05 | $67.95 | 595 |
| rck_alpha_0.5_independent | 2 | 51 | $13.59 | $81.54 | 506 |
| rck_alpha_0.5_independent | 3 | 188 | $48.34 | $132.54 | 587 |
| rck_alpha_0.7_correlated | 1 | 3 | $-16.21 | $83.79 | 625 |
| rck_alpha_0.7_correlated | 2 | 4 | $-0.35 | $83.44 | 546 |
| rck_alpha_0.7_correlated | 3 | 19 | $12.38 | $95.82 | 570 |
| rck_alpha_0.7_independent | 1 | 3 | $-16.21 | $83.79 | 625 |
| rck_alpha_0.7_independent | 2 | 4 | $-0.35 | $83.44 | 546 |
| rck_alpha_0.7_independent | 3 | 19 | $12.34 | $95.78 | 570 |
| rck_alpha_0.8_correlated | 1 | 3 | $-10.86 | $89.14 | 625 |
| rck_alpha_0.8_correlated | 2 | 0 | $0.00 | $89.14 | 575 |
| rck_alpha_0.8_correlated | 3 | 1 | $4.91 | $94.05 | 587 |
| rck_alpha_0.8_independent | 1 | 3 | $-10.86 | $89.14 | 625 |
| rck_alpha_0.8_independent | 2 | 0 | $0.00 | $89.14 | 575 |
| rck_alpha_0.8_independent | 3 | 1 | $4.91 | $94.05 | 587 |

Licznik zmian RCK porownuje najlepsza wykonalna akcje bez ograniczenia (strone i stake) z ostateczna akcja po ograniczeniu; liczy tylko zmiane strony lub przyjetej kwoty.

Kalibracja i estymacja zależności były dopasowane przed każdym blokiem, a etykiety po czasie refit były purge’owane przez `outcome_available_at_utc`. Wyniki są historyczne i mają prior development exposure.

## Artefakty i ograniczenia

- Zamrożona konfiguracja: `configs/research/btc_preopen_advanced_policy_20261007.json`.
- Tabela wyników i manifest: `advanced_policy_study_20261007.json`, `advanced_policy_folds_20261007.csv`, `advanced_policy_trials_20261007.csv`, `advanced_policy_manifest_20261007.json`.
- Pełne decyzje i transakcje pozostają w ignorowanym `data/analysis/polymarket/BTC/preopen_v1/advanced_policy_20261007/`.
- Brak potwierdzonych offline filli ani pełnej drabinki; zapisany best-level jest badawczym modelem wykonania, nie rzeczywistym potwierdzeniem fillu.
- Wynik okresu historycznego nie zastępuje niezależnego okresu po zamrożeniu procedury.
