# Historyczny backtest BTC 5m T−59: fee, wejścia i wyjścia

**Rekomendacja: brak transakcji z powodu braku stabilnej przewagi.** Selekcja wejścia z kontrolą hold-to-resolution wybrała no_trade, bo żaden aktywny wariant wejścia nie przekroczył $100 do 2026-05-31. W późniejszej diagnostyce łączonej polityki wejścia/wyjścia close_at_60s wybrany wyłącznie na treningu podniósł equity challengera do około $140, po czym okres OOS przyniósł około −$139; wynik końcowy spadł do około $1,47. Pełna historia do 2026-10-06 była już wcześniej analizowana w repozytorium i nie jest nietkniętym testem. Modelu nie trenowano, konfiguracji live nie zmieniano i zleceń nie wysyłano.

## Zakres i zamrożone założenia

Badanie objęło 50,184 kwalifikujących się snapshotów T−59 od 2026-04-15 17:05 UTC do 2026-10-06 23:55 UTC. Zamrożono predykcje, model, kalibrator, kotwicę T−59 i zapisane opóźnienie zlecenia 1 s. SHA256 predykcji i artefaktów modelu są zapisane w manifest.json. To kandydat badawczy, nie aktywny bundle live.

Portfel jest jedną ciągłą ścieżką z $100 bez dopłat. Wejście używa ask nie starszego niż 1 s, limitu ceny 0,95, minimum 5 udziałów według obecnego Gamma (historycznego minimum nie udało się potwierdzić) i pełnej żądanej głębokości; brak pełnej głębokości oznacza skip. Fee gotówkowe musi mieścić się w dostępnej gotówce. 5% sizingu liczy się od wolnej gotówki po odjęciu zajętego kosztu. Rozliczenie gotówki następuje 60 s po późniejszym z: oficjalnego resolved_at lub startu rynku + 5 min.

Snapshot T−59 nie gwarantuje fillu po opóźnieniu. Brakuje historycznych ACK, częściowych filli i dokładnych historycznych minimów. Inferencja w replayu miała opóźnienie 0 s; lokalne p50/p95/p99 kandydata wynosi 14,85/22,74/28,00 ms. Opóźnienia sieci, ACK i fillu pozostają niezmierzone. Istniejący replay 521 429 475 zdarzeń nie był powtarzany.

## Historyczne opłaty

Stawka 0,072 jest przypisana do 2026-05-06 00:00 UTC, a 0,07 od tej chwili. Granica 6 maja jest estymacją midpoint między archiwalnymi źródłami, nie potwierdzoną datą zmiany; wrażliwość policzono dla każdego dnia 23 kwietnia–15 maja włącznie i daty nie wybierano według PnL. Opłaty z 14 i 22 kwietnia oraz 4 maja wspierają 0,072; źródła z 8 i 10 maja wspierają 0,07.

Oddzielono formułę od jednostki poboru. Formuła to ilość × stawka × p × (1−p). V1 pobiera fee kupna w udziałach (floor do 6 miejsc na zagregowanym poziomie ceny), a fee sprzedaży w gotówce (floor do 6 miejsc). V2 pobiera gotówkę po obu stronach, zaokrągloną half-up do 5 miejsc z minimum 0,00001. Fee-share obniża udziały/payout; fee gotówkowe zwiększa debet kupna lub zmniejsza wpływ sprzedaży. Fee nie jest podwójnie naliczane. To rekonstrukcja poziomów zagregowanej księgi, nie per-match historyczne filli.

Aktualny zrzut Gamma jest zachowany tylko do audytu i nie służy jako dowód historycznej stawki. Migracja CLOB V1→V2 z 28 kwietnia oraz przybliżone godzinne okno przerwy są modelowane oddzielnie; dokładna sekunda przełączenia pozostaje nieznana. Przypisanie stawki do rynku ma status estymacji, a nie potwierdzenia z jego filli.

## Wejścia i wybór chronologiczny

Przebadano 40 kombinacji: 4 progi edge after-cost (0%, 2%, 4%, 8%) × stałe $5 albo 5% wolnej gotówki z limitami $5/$10/$15/$20/$30/$50/$75/$100/bez limitu. Edge to przewidywany zysk netto podzielony przez debet gotówkowy wejścia. Wybór używa wyłącznie cost-basis equity na koniec treningu 31 maja; żaden wariant aktywny nie pobił $100.

| Wariant | Equity 31 V | Wejścia | Pełny PnL | Gotówka po zwolnieniu | Max DD | Obrót | Fee | Skipy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Brak transakcji (wybór chronologiczny) | $100.00 | 0 | $0.00 | $100.00 | 0.0% | $0.00 | $0.00 | 0 |
| Najlepszy aktywny challenger (diagnoza) | $78.29 | 272 | $-95.16 | $4.84 | 96.2% | $1,360.00 | $58.01 | 49,912 |
| Stałe $5, edge 0% | $0.43 | 683 | $-99.57 | $0.43 | 99.8% | $3,415.00 | $129.16 | 49,501 |
| 5% wolnej gotówki, limit $20, edge 0% | $29.82 | 433 | $-74.98 | $25.02 | 88.9% | $2,681.39 | $101.13 | 49,751 |

Wybrany wariant no-trade utrzymuje $100. Najlepszy aktywny challenger miał equity $78.29 na cutoffie, 217 wejść do cutoffu; jego późniejszy, retrospektywny wycinek OOS od 1 czerwca miał 55 wejść i PnL $-73.45. Stan kapitału jest ciągły, bez resetu w czerwcu.


## Porównanie z poprzednim raportem i wrażliwość fee


W poprzednim raporcie stawka 0,07 była użyta wstecz dla całej historii. Poniżej różnice dla kontroli edge 0%; nowy harmonogram 0,072/0,07, estymowana granica 6 maja i fee-share V1 zmieniają zarówno fee, jak i ścieżkę dostępnego kapitału. Pełne porównanie obrotu, DD i skipów znajduje się w previous_report_comparison.csv.

| Kontrola | Poprzedni PnL | Nowy PnL | Różnica PnL | Poprzednie fee | Nowe fee | Transakcje poprzednio → teraz |
|---|---:|---:|---:|---:|---:|---:|
| fixed_5_usd | $-97.25 | $-99.57 | $-2.32 | $120.77 | $129.16 | 657 → 683 |
| free_cash_5pct_cap20 | $-74.80 | $-74.98 | $-0.18 | $97.80 | $101.13 | 439 → 433 |

Codzienna analiza wrażliwości daty zmiany jest w fee_transition_sensitivity.csv.

## Wyjścia na wspólnej pokrytej próbie

Wyjścia oceniono dla challengera i dwóch kontroli wejścia. Wspólny quote cache obejmuje 963 rynków ze źródłem pełnym; 5 spośród pozyskanych ścieżek nie weszły do paired sample. Hold-to-resolution full-entry control jest osobnym wierszem w tabeli porównań.

| Wejście | Exit | Rynki | Transakcje | Sprzedaże | Hold | PnL | Gotówka końcowa | Fee | Max DD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| challenger edge 8% | hold_to_resolution | 963 | 266 | 0 | 266 | $-97.34 | $2.66 | $56.81 | 97.9% |
| challenger edge 8% | close_at_60s | 963 | 304 | 137 | 167 | $-98.53 | $1.47 | $91.22 | 99.1% |
| challenger edge 8% | close_at_180s | 963 | 152 | 132 | 20 | $-98.69 | $1.31 | $51.30 | 98.8% |
| challenger edge 8% | close_at_240s | 963 | 267 | 134 | 133 | $-98.17 | $1.83 | $71.43 | 98.4% |
| challenger edge 8% | tp10_sl10 | 963 | 293 | 151 | 142 | $-99.13 | $0.87 | $93.68 | 99.4% |
| challenger edge 8% | tp20_sl20 | 963 | 280 | 151 | 129 | $-97.14 | $2.86 | $89.31 | 97.9% |
| stałe $5 edge 0% | hold_to_resolution | 963 | 683 | 0 | 683 | $-99.57 | $0.43 | $129.16 | 99.8% |
| stałe $5 edge 0% | close_at_60s | 963 | 349 | 338 | 11 | $-95.44 | $4.56 | $118.53 | 95.6% |
| stałe $5 edge 0% | close_at_180s | 963 | 339 | 315 | 24 | $-99.20 | $0.80 | $98.86 | 99.4% |
| stałe $5 edge 0% | close_at_240s | 963 | 416 | 389 | 27 | $-95.70 | $4.30 | $113.17 | 97.4% |
| stałe $5 edge 0% | tp10_sl10 | 963 | 284 | 280 | 4 | $-96.55 | $3.45 | $101.43 | 96.6% |
| stałe $5 edge 0% | tp20_sl20 | 963 | 312 | 308 | 4 | $-96.01 | $3.99 | $109.11 | 96.0% |
| 5% cash cap $20 edge 0% | hold_to_resolution | 963 | 433 | 0 | 433 | $-74.98 | $25.02 | $101.13 | 88.9% |
| 5% cash cap $20 edge 0% | close_at_60s | 963 | 337 | 316 | 21 | $-74.55 | $25.45 | $83.67 | 75.2% |
| 5% cash cap $20 edge 0% | close_at_180s | 963 | 283 | 256 | 27 | $-74.92 | $25.08 | $66.92 | 80.6% |
| 5% cash cap $20 edge 0% | close_at_240s | 963 | 339 | 304 | 35 | $-75.36 | $24.64 | $97.19 | 85.9% |
| 5% cash cap $20 edge 0% | tp10_sl10 | 963 | 372 | 356 | 16 | $-75.27 | $24.73 | $82.06 | 75.3% |
| 5% cash cap $20 edge 0% | tp20_sl20 | 963 | 327 | 314 | 13 | $-74.49 | $25.51 | $73.67 | 74.5% |

Źródła quote paths: {'kacho_1hz_sample_proxy': 798, 'pmxt_event_replay': 170}. PMXT używa kolejności receive-time, natywnej głębokości bid i limitu świeżości 1 s; Kacho jest próbką top-of-book 1 Hz z rozmiarem top levelu. Po decyzji obowiązuje 1 s opóźnienia zlecenia. TP/SL ocenia wartość likwidacji netto z fee co 5 s; pełna pozycja musi mieć dostępną głębokość, w przeciwnym razie nie księguje się częściowego wyjścia. Progi można ponawiać na kolejnych przyczynowych kwotowaniach; stały close ma jedną próbę. Sprzedaż uwalnia środki dopiero w chwili wykonania, a wejście o tym samym timestamp jest obsługiwane wcześniej.

Pomiar wykonania exit study: 968 quote paths, selektywna ekstrakcja 0.00 s, discovery simulation 11.42 s, cache hits 968; próbka 12 rynków (6 PMXT, 6 Kacho) zajęła 1.78 s. Pierwsze przejście cold-cache: 968 ścieżek w 9.29 s. Cały run: 27.89 s; peak working set: 2416.3 MiB (0 oznacza brak pomiaru).

Walk-forward exit-only na challengerze wybrał do 31 maja close_at_60s: equity $140.05 wobec $78.29 dla hold. W OOS ta polityka wykonała 87 wejść, miała PnL $-138.58 i końcowe equity $1.47. Ta niestabilność uzasadnia ogólną rekomendację no-trade.

## Rabaty

Obecna dokumentacja podaje start programu taker 28 maja 2026. Dla challenger + close_at_60s w common-sample ledger po starcie programu oszacowano 138 wejść i 0 sprzedaży; wariantowa estymacja wypłaconej gotówki to $0.00, a bonusów $0.00.
To wyłącznie overlay według aktualnie zapisanych progów. Historyczne warunki launchu nie są potwierdzone; ponieważ strona nie podaje pieniężnej podstawy procentu rabatu, estymacja mnoży go przez zrekonstruowane fee wykonanych transakcji taker i jest ilustracyjna. Dla sprzedaży samo historyczne uprawnienie do rabatu również jest niepotwierdzone. Wartość wV używa formuły rozmiar × (1−cena) × waga crypto 2,3. Dzienna zmiana tieru i wypłata o północy UTC z minimum $1 są modelowane oddzielnie; naliczone, niewypłacone środki nie finansują wcześniejszych wejść. Bonusy i rabaty nie wpływają na wybór; maker rebate nie jest naliczany.

## Pliki i źródła

- historical_policy_report.md — to podsumowanie.
- policy_comparison.csv, monthly_portfolio.csv, finalist_trade_ledger.csv — pełna siatka wejść, miesięczne stany i ledger.
- reports/btc_fee_history_20261010/independent_signal_diagnostic_summary.csv — niezależnie finansowana diagnostyka zachowana osobno, nie jest wynikiem portfela $100.
- fee_transition_sensitivity.csv, fee_rule_assignment_audit.csv.gz — zmiana daty fee oraz przypisanie stawki/jednostki.
- exit_policy_comparison.csv, exit_policy_monthly.csv, exit_trade_ledger.csv, exit_path_coverage.csv — wyjścia i wspólne quote paths.
- taker_rebate_estimate.csv, taker_rebate_cashflows_estimate.csv, taker_rebate_trade_ledger_estimate.csv — odseparowany szacunek aktualnych warunków.
- manifest.json, walk_forward_selection.json, exit_study_manifest.json, taker_rebate_estimate_manifest.json — pochodzenie i założenia.
- sources/ oraz source_provenance.json — zachowane kopie źródeł i ich SHA256.

Archiwalne źródła: Polymarket Fees z 14 i 22 kwietnia, Maker Rebates z 4 maja, Fees z 8 i 10 maja, informacja o migracji z 28 kwietnia oraz aktualna strona Taker Rebates. Ich URL-e, kopie i skróty SHA256 są w source_provenance.json.
