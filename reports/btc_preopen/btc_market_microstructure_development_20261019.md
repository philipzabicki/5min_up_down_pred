# Development: RAW – rynek – mikrostruktura — 2026-10-19

## Zakres i zamrożona korekta protokołu

To eksploracyjny test development z hipotezą pochodzącą z wcześniejszych analiz. Holdout nie był odczytywany ani oceniany. Wersję 2 zapisano przed fitami: kontrola C ma 7 wejść, a challenger X ma te same wejścia, d = logit(p_RAW) − logit(p_market) i sześć ustalonych interakcji d × z_j (łącznie 14). Addytywnej wersji z projektu nie dopasowano.

Oba modele korzystają ze wspólnego StandardScaler dopasowanego na siedmiu bazowych wejściach i d wyłącznie na treningu. Interakcje to iloczyny wystandaryzowanych d i z_j; osobny scaler interakcji jest również dopasowany tylko na treningu. Stałe kolumny zachowano, a StandardScaler mapuje je na zero. Wspólne siedem wejść C i X jest identyczne w treningu i ocenie.

Fit: jeden prefiks od 15 kwietnia do startów przed 15 maja; release etykiety musi poprzedzać pierwszą decyzję testową T−60 (2026-05-14T23:59:00+00:00). Ocena obejmuje starty od 15 maja do końca developmentu, bez refitów. Wymagane minimum to 1 000 etykiet i obie klasy.

## Audyt dostępności bez etykiet

Lejek zaczyna się od 21,970 wierszy RAW OOF; 21,961 ma dopasowany snapshot T−60, a 7,010 przechodzi wszystkie warunki sześciu cech. Ponowne przeliczenie buildera zgadza się z zapisanym feature table pod względem indeksów i wartości.

### Kolejne wykluczenia

| Warunek | Przed | Wykluczono | Pozostało |
|---|---|---|---|
| raw_valid | 21,970 | 0 | 21,970 |
| snapshot_match | 21,970 | 9 | 21,961 |
| t60_aligned | 21,961 | 0 | 21,961 |
| source_event_observed | 21,961 | 5,364 | 16,597 |
| source_coverage_chronology | 16,597 | 45 | 16,552 |
| native_both_books | 16,552 | 2 | 16,550 |
| unambiguous_asks | 16,550 | 7,265 | 9,285 |
| not_crossed | 9,285 | 2,275 | 7,010 |
| p_market_valid | 7,010 | 0 | 7,010 |
| ask_ladder_aligned | 7,010 | 0 | 7,010 |
| gross_depth_5_both_sides | 7,010 | 0 | 7,010 |
| six_features_finite | 7,010 | 0 | 7,010 |

### Niezależne częstości flag

| Flaga | Prawda | Kohorta |
|---|---|---|
| raw_probability_valid | 21,970 | 21,970 |
| matching_t60_snapshot | 21,961 | 21,970 |
| exact_t60_alignment | 21,961 | 21,970 |
| at_least_one_source_event_in_replay | 16,597 | 21,970 |
| archive_entry_hour_available | 21,925 | 21,970 |
| archive_freshness_window_available | 21,925 | 21,970 |
| no_future_receive_event | 21,961 | 21,970 |
| no_future_source_event | 21,961 | 21,970 |
| up_gap_after_init_flag_true | 45 | 21,970 |
| down_gap_after_init_flag_true | 44 | 21,970 |
| source_coverage_and_chronology_pass | 21,916 | 21,970 |
| full_two_book_snapshot | 16,595 | 21,970 |
| both_books_native | 16,595 | 21,970 |
| up_quote_complemented | 1 | 21,970 |
| down_quote_complemented | 0 | 21,970 |
| up_ask_ambiguous | 5,321 | 21,970 |
| down_ask_ambiguous | 4,093 | 21,970 |
| up_book_crossed | 5,733 | 21,970 |
| down_book_crossed | 5,336 | 21,970 |
| up_book_locked | 2,677 | 21,970 |
| down_book_locked | 2,788 | 21,970 |
| quote_valid_flag | 10,839 | 21,970 |
| valid_p_market | 10,794 | 21,970 |
| up_best_ask_matches_first_level | 16,596 | 21,970 |
| down_best_ask_matches_first_level | 16,596 | 21,970 |
| gross_5usd_depth_up | 16,596 | 21,970 |
| gross_5usd_depth_down | 16,596 | 21,970 |
| saved_fee_adjusted_5usd_depth_flag | 16,476 | 21,970 |
| all_six_features_finite | 7,010 | 21,970 |

Dzienny lejek dla pełnego kalendarza 15 kwietnia–30 czerwca znajduje się w data/analysis/polymarket/BTC/btc_market_microstructure_development_20261019/availability/daily_funnel.csv. Poniżej pokazano dzienne liczebności kluczowych etapów.

| Dzień UTC | RAW | Snapshot T−60 | Zdarzenia | Pokrycie/chronologia | Dwie księgi | Ask jednoznaczne | Nie crossed | p_market | Końcowe |
|---|---|---|---|---|---|---|---|---|---|
| 2026-04-15 00:00:00+00:00 | 83 | 83 | 83 | 83 | 83 | 42 | 42 | 42 | 19 |
| 2026-04-16 00:00:00+00:00 | 288 | 286 | 286 | 286 | 286 | 135 | 141 | 141 | 68 |
| 2026-04-17 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 114 | 176 | 176 | 65 |
| 2026-04-18 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 103 | 157 | 157 | 62 |
| 2026-04-19 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 84 | 130 | 130 | 38 |
| 2026-04-20 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 111 | 148 | 148 | 58 |
| 2026-04-21 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 119 | 157 | 157 | 68 |
| 2026-04-22 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 158 | 145 | 145 | 88 |
| 2026-04-23 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 169 | 132 | 132 | 81 |
| 2026-04-24 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 146 | 157 | 157 | 84 |
| 2026-04-25 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 149 | 109 | 109 | 71 |
| 2026-04-26 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 107 | 106 | 106 | 48 |
| 2026-04-27 00:00:00+00:00 | 288 | 288 | 288 | 288 | 287 | 25 | 118 | 118 | 11 |
| 2026-04-28 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 67 | 140 | 140 | 36 |
| 2026-04-29 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 120 | 116 | 116 | 53 |
| 2026-04-30 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 116 | 145 | 145 | 61 |
| 2026-05-01 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 142 | 122 | 122 | 61 |
| 2026-05-02 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 104 | 151 | 151 | 54 |
| 2026-05-03 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 183 | 118 | 118 | 79 |
| 2026-05-04 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 162 | 159 | 159 | 97 |
| 2026-05-05 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 118 | 139 | 139 | 49 |
| 2026-05-06 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 148 | 167 | 167 | 90 |
| 2026-05-07 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 154 | 177 | 177 | 99 |
| 2026-05-08 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 91 | 134 | 134 | 42 |
| 2026-05-09 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 96 | 155 | 155 | 52 |
| 2026-05-10 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 121 | 105 | 105 | 43 |
| 2026-05-11 00:00:00+00:00 | 288 | 288 | 288 | 288 | 287 | 157 | 135 | 134 | 78 |
| 2026-05-12 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 213 | 138 | 138 | 106 |
| 2026-05-13 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 170 | 121 | 121 | 77 |
| 2026-05-14 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 189 | 143 | 143 | 89 |
| 2026-05-15 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 228 | 203 | 203 | 161 |
| 2026-05-16 00:00:00+00:00 | 288 | 288 | 285 | 288 | 285 | 162 | 137 | 134 | 69 |
| 2026-05-17 00:00:00+00:00 | 288 | 288 | 286 | 288 | 286 | 148 | 104 | 102 | 54 |
| 2026-05-18 00:00:00+00:00 | 288 | 288 | 286 | 288 | 286 | 74 | 133 | 131 | 32 |
| 2026-05-19 00:00:00+00:00 | 288 | 288 | 285 | 288 | 285 | 33 | 170 | 167 | 22 |
| 2026-05-20 00:00:00+00:00 | 288 | 288 | 285 | 288 | 285 | 75 | 142 | 139 | 36 |
| 2026-05-21 00:00:00+00:00 | 288 | 288 | 181 | 288 | 181 | 173 | 190 | 83 | 28 |
| 2026-05-22 00:00:00+00:00 | 288 | 288 | 91 | 288 | 91 | 263 | 285 | 88 | 66 |
| 2026-05-23 00:00:00+00:00 | 288 | 288 | 287 | 288 | 287 | 123 | 208 | 207 | 105 |
| 2026-05-24 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 2 | 152 | 152 | 1 |
| 2026-05-25 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 21 | 205 | 205 | 20 |
| 2026-05-26 00:00:00+00:00 | 288 | 287 | 287 | 287 | 287 | 185 | 239 | 239 | 185 |
| 2026-05-27 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 277 | 288 | 288 | 277 |
| 2026-05-28 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 270 | 288 | 288 | 270 |
| 2026-05-29 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 259 | 288 | 288 | 259 |
| 2026-05-30 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 279 | 288 | 288 | 279 |
| 2026-05-31 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 280 | 288 | 288 | 280 |
| 2026-06-01 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 280 | 288 | 288 | 280 |
| 2026-06-02 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 282 | 288 | 288 | 282 |
| 2026-06-03 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 259 | 288 | 288 | 259 |
| 2026-06-04 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 275 | 288 | 288 | 275 |
| 2026-06-05 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 272 | 288 | 288 | 272 |
| 2026-06-06 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 261 | 288 | 288 | 261 |
| 2026-06-07 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 262 | 288 | 288 | 262 |
| 2026-06-08 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 250 | 288 | 288 | 250 |
| 2026-06-09 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 264 | 288 | 288 | 264 |
| 2026-06-10 00:00:00+00:00 | 288 | 288 | 288 | 288 | 288 | 257 | 288 | 288 | 257 |
| 2026-06-11 00:00:00+00:00 | 288 | 288 | 288 | 243 | 288 | 261 | 288 | 243 | 219 |
| 2026-06-12 00:00:00+00:00 | 288 | 288 | 41 | 288 | 41 | 287 | 288 | 41 | 40 |
| 2026-06-13 00:00:00+00:00 | 288 | 288 | 0 | 288 | 0 | 288 | 288 | 0 | 0 |
| 2026-06-14 00:00:00+00:00 | 288 | 288 | 0 | 288 | 0 | 288 | 288 | 0 | 0 |
| 2026-06-15 00:00:00+00:00 | 288 | 288 | 0 | 288 | 0 | 288 | 288 | 0 | 0 |
| 2026-06-16 00:00:00+00:00 | 288 | 288 | 0 | 288 | 0 | 288 | 288 | 0 | 0 |
| 2026-06-17 00:00:00+00:00 | 288 | 283 | 0 | 283 | 0 | 283 | 283 | 0 | 0 |
| 2026-06-18 00:00:00+00:00 | 288 | 288 | 0 | 288 | 0 | 288 | 288 | 0 | 0 |
| 2026-06-19 00:00:00+00:00 | 288 | 288 | 0 | 288 | 0 | 288 | 288 | 0 | 0 |
| 2026-06-20 00:00:00+00:00 | 288 | 287 | 0 | 287 | 0 | 287 | 287 | 0 | 0 |
| 2026-06-21 00:00:00+00:00 | 288 | 288 | 0 | 288 | 0 | 288 | 288 | 0 | 0 |
| 2026-06-22 00:00:00+00:00 | 288 | 288 | 0 | 288 | 0 | 288 | 288 | 0 | 0 |
| 2026-06-23 00:00:00+00:00 | 288 | 288 | 0 | 288 | 0 | 288 | 288 | 0 | 0 |
| 2026-06-24 00:00:00+00:00 | 288 | 288 | 0 | 288 | 0 | 288 | 288 | 0 | 0 |
| 2026-06-25 00:00:00+00:00 | 288 | 288 | 0 | 288 | 0 | 288 | 288 | 0 | 0 |
| 2026-06-26 00:00:00+00:00 | 288 | 288 | 0 | 288 | 0 | 288 | 288 | 0 | 0 |
| 2026-06-27 00:00:00+00:00 | 288 | 288 | 0 | 288 | 0 | 288 | 288 | 0 | 0 |
| 2026-06-28 00:00:00+00:00 | 288 | 288 | 0 | 288 | 0 | 288 | 288 | 0 | 0 |
| 2026-06-29 00:00:00+00:00 | 288 | 288 | 91 | 288 | 91 | 198 | 274 | 77 | 1 |
| 2026-06-30 00:00:00+00:00 | 287 | 287 | 287 | 287 | 287 | 17 | 181 | 181 | 17 |

Flagi dostępności godziny i freshness oznaczają, że partycja została odnaleziona lub przetworzona; nie dowodzą, że zawiera zdarzenia. Obecność zdarzeń sprawdzono osobno.

Lokalne partycje `event_parts` dla 384 godzin od 13 do 28 czerwca mają 0 wierszy po selekcji rynku i czasu. Checkpoint odnotowuje odczyt grup Parquet, ale oryginalne bajty źródłowe nie są lokalne, a liczby wierszy przed filtrem nie zapisano. Snapshoty T−60 zawierają 4 602 rekordy z `market_event_count = 0`; sześć startów nie ma rekordu Gamma. **Przyczyna nieustalona z lokalnych artefaktów** — pusty wynik filtrowania nie dowodzi pustego oryginalnego archiwum ani nie wyklucza błędu selekcji. Szczegóły i tabela dowodów: [audyt pochodzenia pustych partycji](btc_empty_partition_provenance_20261019.md).

Kod rekonstrukcji grupuje zdarzenia po czasie odbioru i pilnuje porządku czasu źródłowego w obrębie tokenu i strony. Crossing oznacza best_bid > best_ask, locked oznacza równość; spóźnione zmiany cross-side są wykrywane i stosowane, jeśli zachowują porządek danej strony. Sam crossing nie dowodzi błędu. Istniejący quote_valid odrzuca crossed books; walidacji nie poluzowano. Niezależnie crossing wystąpił 5,733 razy dla UP i 5,336 dla DOWN.

Flaga depth_5usd_valid_both_sides z replayu kontroluje wykonanie wraz z dostępnością historycznej opłaty. Cechy slippage sprawdzono niezależnie na zapisanych drabinkach brutto: poziomy muszą obsłużyć pełne $5. Opłaty zachowano jako osobną część audytu ekonomii.

## Dopasowanie i jakość prognoz

Trening obejmował 1,927 wierszy, klasy {'0': 961, '1': 966}; ostatni release etykiety nastąpił 2026-05-14T23:51:21+00:00. Ocena ma 5,083 wierszy zakwalifikowanych wyłącznie po wejściach. Do oceny jakości release do cutoffu uzyskano 5,082 etykiet z 31 dni. Wspólna próba obejmuje też końcówkę czerwca.

| Wariant | N | Dni z obserwacjami | Log-loss | Brier | AUC |
|---|---|---|---|---|---|
| MARKET | 5,082 | 31 | 0.693141 | 0.249990 | 0.517883 |
| RAW | 5,082 | 31 | 0.692210 | 0.249532 | 0.526353 |
| C | 5,082 | 31 | 0.699754 | 0.252686 | 0.519284 |
| X | 5,082 | 31 | 0.711804 | 0.254542 | 0.518909 |

Miesięczne wyniki na tej samej próbie:

| Miesiąc UTC | Wariant | N | Dni | Log-loss | Brier | AUC |
|---|---|---|---|---|---|---|
| 2026-05 | MARKET | 2,143 | 17 | 0.693072 | 0.249953 | 0.516107 |
| 2026-05 | RAW | 2,143 | 17 | 0.692470 | 0.249662 | 0.516344 |
| 2026-05 | C | 2,143 | 17 | 0.702722 | 0.253648 | 0.511574 |
| 2026-05 | X | 2,143 | 17 | 0.723814 | 0.257210 | 0.509090 |
| 2026-06 | MARKET | 2,939 | 14 | 0.693192 | 0.250017 | 0.515336 |
| 2026-06 | RAW | 2,939 | 14 | 0.692021 | 0.249438 | 0.538751 |
| 2026-06 | C | 2,939 | 14 | 0.697590 | 0.251985 | 0.523115 |
| 2026-06 | X | 2,939 | 14 | 0.703046 | 0.252596 | 0.526991 |

Sparowane 95% przedziały bootstrapu bloków 3-dniowych (ujemna różnica log-loss/Brier oznacza poprawę po lewej stronie):

| Kontrast | Log-loss 95% CI | Ważne | Brier 95% CI | Ważne | AUC 95% CI | Ważne |
|---|---|---|---|---|---|---|
| X_MINUS_C | [0.002681, 0.025441] | 2,000 | [0.000720, 0.003371] | 2,000 | [-0.011273, 0.009873] | 2,000 |
| C_MINUS_MARKET | [0.002171, 0.011579] | 2,000 | [0.000896, 0.004571] | 2,000 | [-0.012737, 0.011336] | 2,000 |
| X_MINUS_MARKET | [0.006281, 0.036709] | 2,000 | [0.002101, 0.007682] | 2,000 | [-0.016032, 0.012421] | 2,000 |

Bootstrap użył pełnego kalendarza 47 dni od 15 maja do 30 czerwca, w tym dni bez kwalifikujących się wierszy. Replikacje: 2,000; puste: 0; jednoklasowe: 0 (zachowane dla log-loss/Brier, AUC oznaczone jako niezdefiniowane). Głównym kontrastem jest X−C; X−MARKET porównuje challenger bezpośrednio z benchmarkiem rynku.

## Ekonomia na wspólnym kalendarzu

Każdy wariant podejmuje własne decyzje na tych samych wejściowych wierszach. Warianty C, X i MARKET używają wyłącznie T−60 do wyboru strony i progu EV netto co najmniej 1%; snapshot T−59 służy do wykonania wybranej strony. RAW pochodzi z istniejącego ledgeru ograniczonego najpierw do wspólnego kalendarza. Wyniki po cutoffie pozostają nierozliczone.

| Wariant | Decyzje | EV ≥1% | Wykonania | Rozliczono | Nierozliczone | Notional settled $ | Debet settled $ | Opłata cash $ | Opłata tokenowa (shares) | PnL $ | PnL/notional | PnL/debet | EV na settled $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C | 5,083 | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | — | — | 0.000000 |
| X | 5,083 | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | — | — | 0.000000 |
| MARKET | 5,083 | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | — | — | 0.000000 |
| RAW | 5,083 | 787 | 782 | 782 | 0 | 3910.000000 | 4051.919970 | 141.919970 | 0.000000 | 229.603416 | 0.058722 | 0.056665 | 207.510178 |

Notional, debet, PnL i EV dotyczą tych samych rozliczonych wykonań. Opłaty gotówkowe pozostają w debecie; opłaty pobrane w outcome shares zmniejszają liczbę wypłaconych tokenów i nie są doliczane do cash debit. Liczby otwartych pozycji i ich debet są również w manifeście i ledgerach.

Wyniki miesięczne:

| Miesiąc UTC | Wariant | Decyzje | EV ≥1% | Wykonania | Rozliczono | Nierozliczone | Notional $ | Debet $ | Fee cash $ | Fee shares | PnL $ | PnL/notional | PnL/debet | EV settled $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-05 | C | 2,145 | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | — | — | 0.000000 |
| 2026-06 | C | 2,938 | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | — | — | 0.000000 |
| 2026-05 | X | 2,145 | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | — | — | 0.000000 |
| 2026-06 | X | 2,938 | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | — | — | 0.000000 |
| 2026-05 | MARKET | 2,145 | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | — | — | 0.000000 |
| 2026-06 | MARKET | 2,938 | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | — | — | 0.000000 |
| 2026-05 | RAW | 2,145 | 271 | 267 | 267 | 0 | 1335.000000 | 1385.645170 | 50.645170 | 0.000000 | 11.129747 | 0.008337 | 0.008032 | 116.235041 |
| 2026-06 | RAW | 2,938 | 516 | 515 | 515 | 0 | 2575.000000 | 2666.274800 | 91.274800 | 0.000000 | 218.473669 | 0.084844 | 0.081940 | 91.275136 |

Dzienna różnica PnL X−C obejmuje wszystkie 47 dni, z zerami w dniach bez transakcji. Średnia dzienna: 0.000000 USD, 95% CI [0.000000, 0.000000]; różnica sumy: 0.000000 USD, 95% CI [0.000000, 0.000000].

Koncentracja dzienna jest zapisana w manifeście. Udział PnL z pięciu najlepszych dni: C —, X —, MARKET —, RAW 0.966720. Pełne dzienne szeregi: data/analysis/polymarket/BTC/btc_market_microstructure_development_20261019/economics/daily_pnl.csv.

## Współczynniki i interpretacja

Współczynniki X są podane w skali cech po transformacjach zapisanych z treningu. Interakcje są opisowe i nie stanowią osobno potwierdzonych efektów.

| Interakcja | Współczynnik X |
|---|---|
| raw_minus_market_logit_x_mid_sum_minus_one | -0.104918 |
| raw_minus_market_logit_x_slippage_5_down | 0.035005 |
| raw_minus_market_logit_x_slippage_5_up | -0.023745 |
| raw_minus_market_logit_x_spread_down | -0.241513 |
| raw_minus_market_logit_x_spread_up | 0.085070 |
| raw_minus_market_logit_x_up_down_ask_depth_imbalance | -0.072116 |

Wniosek: X nie wnosi potwierdzonej informacji BTC ponad C w tym teście: różnica log-loss X−C wynosi 0.012049, a jej sparowany 95% CI to [0.002681, 0.025441]. X wypada też gorzej od MARKET: X−MARKET = 0.018662, 95% CI [0.006281, 0.036709].
Nie widać dodatniej ekonomii dla C, X ani MARKET: liczba zleceń z EV co najmniej 1% wyniosła odpowiednio 0, 0 i 0; żaden z tych wariantów nie miał wykonań. RAW z istniejącego ledgeru na tym samym kalendarzu miał 782 wykonań i PnL 229.60 USD po opłatach; ten wynik nie pochodzi z X.
Zakres wniosku ogranicza luka 16 dni bez zdarzeń źródłowych oraz 18 wejść po 12 czerwca. Wynik nie opisuje stabilności w tej luce ani na holdoucie.

## Koszt i artefakty

Czas całkowity: 31.08 s; szczytowy working set procesu: 1,513,316,352 B. Python 3.14.4; sklearn 1.8.0; CPU logiczne 20; fit ograniczony do jednego wątku BLAS. Modele, scalery i predykcje zapisano przed bootstrapem i raportowaniem; checkpoint: data/analysis/polymarket/BTC/btc_market_microstructure_development_20261019/pre_bootstrap_checkpoint.json.

Raport: reports/btc_preopen/btc_market_microstructure_development_20261019.md. Manifest: reports/btc_preopen/btc_market_microstructure_development_20261019_manifest.json. Dane, modele i checkpointy pozostają poza commitem; wersjonowane są protokół, kod, testy, raport i manifest.
