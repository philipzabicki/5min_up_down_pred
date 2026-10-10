# BTC Polymarket 5m — rekonstrukcja reguł opłat i przeliczenie T−59

**Zakres:** zapisane snapshoty T−59 od 2026-04-15 17:05 UTC do 2026-10-06 23:55 UTC. Użyłem gotowego replayu i drabinek; nie odtwarzałem archiwum 521 mln zdarzeń, nie trenowałem modelu i nie wysyłałem zleceń.

## Rejestr reguł rynków BTC 5m

Pobrałem bieżące pola Gamma `feesEnabled` i `feeSchedule` osobno dla każdego z 79 680 potwierdzonych rynków z zachowanego kalendarza, po `condition_id`, korzystając z [Gamma Markets API](https://gamma-api.polymarket.com/openapi.json) i jego [metadanych rynku](https://docs.polymarket.com/market-data/market-details); brak jednego rekordu w paczce uzupełniłem zapytaniem po slugu. Zapisano je w skompresowanym eksporcie z czasem przechwycenia 2026-10-10 UTC. Wszystkie 50 184 rynki z wejściami T−59 mają `feesEnabled=true`, `rate=0.07`, `exponent=1`, `takerOnly=true`, `feeType=crypto_fees_v2`.

| Stawka Gamma | Wykładnik | feeType | Rynki w kalendarzu |
|---:|---:|---|---:|
| `0.07` | `1.0` | `crypto_fees_v2` | 55,008 |
| `—` | `—` | `—` | 11,431 |
| `0.25` | `2.0` | `crypto_fees` | 7,562 |
| `0.25` | `2.0` | `crypto_15_min` | 5,679 |

Reguły na rynkach BTC 5m: `feesEnabled=false` dla 11 431 rynków od 2025-12-18 04:25 do 2026-01-26 23:20 UTC; kalendarz nie ma rynków od 2026-01-26 23:25 do 2026-02-12 00:30 UTC; pierwszy aktywny rekord to 2026-02-12 00:35 UTC z `0.25, exponent=2`; ostatni taki rekord kończy się na rynku 2026-03-29 23:55 UTC. Rynek 2026-03-30 00:00 UTC jest pierwszym z `0.07, exponent=1`, który utrzymuje się do końca kalendarza. Przykłady granicy: [ostatni rynek 0.25](https://gamma-api.polymarket.com/markets/slug/btc-updown-5m-1774828500) i [pierwszy rynek 0.07](https://gamma-api.polymarket.com/markets/slug/btc-updown-5m-1774828800). Zmiana `feeType` w obrębie `0.25, exponent=2` nie zmieniła krzywej opłaty. Stawki pochodzą z metadanych rzeczywistych rynków BTC 5m; nie przeniosłem taryfy z rynków 15m.

Granica w pustym przedziale stycznia/lutego nie ma wpływu na żaden rynek, bo wtedy nie ma rynku BTC 5m w kalendarzu. Metadane Gamma zostały pobrane 2026-10-10; archiwum nie zachowało oryginalnych pól `feeSchedule` przy utworzeniu każdego rynku. Rejestr pozostawia tę różnicę źródłową widoczną.

## Pobór opłaty i pole archiwalne

W starszym CTF Exchange kupujący płacił w udziałach; od migracji z 28 kwietnia opłata jest pobierana w USDC/pUSD przy matchu, zgodnie z [komunikatem migracyjnym](https://help.polymarket.com/en/articles/14762452-polymarket-exchange-upgrade-april-28-2026). Publiczny komunikat podaje przybliżone okno 11:00–12:00 UTC, więc symulacja nie handluje w tym oknie. Regułę taryfy dobieram z metadanych konkretnego rynku, a jednostkę poboru z czasu wejścia/matchu.

| `last_trade_price.fee_rate_bps` | Snapshoty | Interpretacja |
|---|---:|---|
| jawne zero | 37,961 | pole archiwalne ma wartość 0; po V2 nie dowodzi darmowej transakcji |
| 1000 bps | 3,674 | maksymalny limit `feeRateBps` starego zlecenia; kontrakt i moduł zwracają niewykorzystany limit, więc to nie kwota pobrana |
| brak stawki | 8,549 | unknown; nie zastąpiono zerem |

Dawny [CTF fee calculator](https://github.com/Polymarket/ctf-exchange/blob/main/src/exchange/libraries/CalculatorHelper.sol) ustala limit opłaty z `feeRateBps`, a [FeeModule](https://github.com/Polymarket/exchange-fee-module/blob/main/src/FeeModule.sol) zwraca nadmiar ponad opłatę przekazaną przez operatora. Archiwizer zachowuje brak jako null, więc 37 961 zer to jawne wartości w payloadzie; jednak nie są dowodem darmowego rynku, bo wszystkie te rynki mają w Gamma aktywne `0.07`. Dlatego ani `1000`, ani archiwalne zero nie są użyte jako taryfa. [Aktualne dokumenty](https://docs.polymarket.com/trading/fees) i [oficjalny SDK](https://github.com/Polymarket/py-clob-client-v2/blob/main/py_clob_client_v2/fees.py) opisują `fee = shares × rate × (p × (1-p))^exponent`; dla poboru gotówkowego model zaokrągla sumę na zleceniu do 5 miejsc i stosuje minimum $0.00001.

**Ograniczenie fill’i:** Gamma potwierdza stawkę i wykładnik konkretnego rynku, a komunikat migracyjny potwierdza zmianę jednostki poboru. Zapisane ask ladder ma zagregowaną głębokość po poziomie ceny, nie identyfikatory zleceń makerów ani historyczne fee kwoty per match. Dla V1 pokazuję wyraźnie oznaczoną estymację: udokumentowaną formułę taryfy przeliczam na udziały i zaokrąglam w dół do 6 miejsc na zagregowany poziom ceny. Ten rachunek nie jest potwierdzonym księgowaniem V1. W V2 potwierdzona jest reguła gotówkowa, a raport zaokrągla sumę modelowanego zlecenia; dokładne historyczne opłaty per match nadal są nieobserwowane. Oba segmenty opisują symulowane zlecenia na snapshotach, nie faktycznie wykonane transakcje. Rebate nie jest odejmowany: symulowane zakupy są takerami, a `takerOnly=true`.

## Wynik portfeli

Każdy scenariusz startuje z $100, używa stałego $5 albo `min(5% wolnej gotówki, cap)` dla capów $5/$10/$15/$20/$30/$50/$75/$100/brak limitu, tego samego porządku rozliczeń z 60 s zwłoki, pełnych zapisanych drabinek i limitu ceny 0.95. Tabela skraca wyniki do stałego $5 i capu $20; `scenario_summary.csv` ma pełną siatkę, w tym świeżość 1/5/30 s.

| Segment | Polityka | Transakcje | Opłaty estymowane | PnL netto | Kapitał końcowy | Max DD | Pominięcia |
|---|---|---:|---:|---:|---:|---:|---:|
| pełna ścieżka, taryfa historyczna modelowana | `fixed_5_usd` | 657 | $120.77 | $-97.25 | $2.75 | 98.70% | 49,527 |
| pełna ścieżka, taryfa historyczna modelowana | `free_cash_5pct_cap20` | 439 | $97.80 | $-74.80 | $25.20 | 89.11% | 49,745 |
| segment potwierdzonej jednostki poboru V2, start $100 | `fixed_5_usd` | 284 | $52.38 | $-95.03 | $4.97 | 97.05% | 46,218 |
| segment potwierdzonej jednostki poboru V2, start $100 | `free_cash_5pct_cap20` | 287 | $54.78 | $-74.92 | $25.08 | 87.06% | 46,215 |

Ciągła ścieżka obejmuje 50,184 rynków: 3,670 wejść przed migracją z poborem w udziałach, 12 wejść z okna konserwacji i 46,502 wejść z poborem gotówkowym. Nierozstrzygnięta kwota V1 może wpływać na wybór strony, stawki, późniejszą gotówkę i reinwestowanie; dlatego tej ścieżki nie opisuję jako w pełni potwierdzonej. Potwierdzony segment zaczyna się przy pierwszym wejściu po 2026-04-28 12:00 UTC i resetuje kapitał do $100. Liczba nierozstrzygniętych reguł rynkowych w badanej próbie: 0.

Scenariusze z kapitałem końcowym równym zero: 0. To nie oznacza, że każdy portfel mógł dalej handlować: pierwsze pominięcie fixed-$5 z powodu niewystarczającej wolnej gotówki uruchamia poniższą osobną diagnostykę sygnałów.

### Diagnostyka sygnałów po ograniczeniu kapitałem

Diagnostyka zaczyna się od pierwszego wejścia, które portfel fixed-$5 pominął wyłącznie z powodu dostępnej gotówki (to wejście jest włączone). Dalej używa tych samych filtrów, taryf, ladderów i stawki $5, ale ma niezależną rezerwę, więc wynik pokazuje sygnały po ograniczeniu kapitałem, a nie wykonalny portfel startujący ze $100. PnL jest sumą hipotetycznych wyników tych sygnałów.

| Segment | Pierwszy sygnał ograniczony gotówką UTC | Wykonane sygnały $5 | Opłaty estymowane | PnL sygnałów | Pominięcia z innych filtrów |
|---|---|---:|---:|---:|---:|
| `continuous_mixed_regime_estimate` | 2026-05-03T06:24:01+00:00 | 7,183 | $1297.77 | $-430.23 | 37,947 |
| `post_upgrade_confirmed_fee_segment` | 2026-05-05T22:19:01+00:00 | 7,086 | $1279.92 | $-355.84 | 37,277 |

Szczegóły każdego wejścia tej diagnostyki są w `independent_signal_diagnostic_ledger.csv.gz`; rezerwa finansująca służy wyłącznie do usunięcia limitu gotówki i nie jest traktowana jako kapitał portfela.

## Wpływ na wybór i porównanie z poprzednimi etykietami

| Polityka | Obecny 0.07 cash counterfactual: transakcje | Nowa estymacja: transakcje | Wspólne wejścia | Zmieniona strona na wspólnych wejściach | Mediana ΔVWAP |
|---|---:|---:|---:|---:|---:|
| `fixed_5_usd` | 457 | 657 | 439 | 0 | $0.00000 |
| `free_cash_5pct_cap20` | 417 | 439 | 397 | 0 | $0.00000 |

Porównanie jest ścieżkowe, a nie addytywna dekompozycja: opłata wpływa na EV i wybór strony, a wcześniejsze wyniki zmieniają wolną gotówkę, wielkość następnej stawki i dostępność późniejszych wejść. `counterfactual_comparison.csv` zestawia też liczbę transakcji, obrót, opłaty, PnL, drawdown i pominięcia z zachowanymi scenariuszami starego raportu.

Stare raporty pozostają w archiwum pod trzema czytelnymi etykietami: `archived_event_bps_literal` = literalna wartość pola zdarzenia; `current_gamma_schedule_0p07_counterfactual` = obecna taryfa zastosowana do całej historii; `continuous_mixed_regime_estimate` = odtworzone taryfy rynku z estymacją V1; `post_upgrade_confirmed_fee_segment` = potwierdzona taryfa gotówkowa od 28 kwietnia, ze świeżym saldem $100.

## Zakres pewności i artefakty

Wyniki dotyczą historycznej ekonomiki zapisanych snapshotów, nie gotowości live. Nie dziedziczą automatycznie wcześniejszego NO-GO operacyjnego; ta analiza nie mierzy ACK, częściowych fill’i ani rzeczywiście pobranych fee. Post-V2 segment stosuje potwierdzoną regułę, ale opłaty i PnL nadal są estymacją z historycznej drabinki, nie zapisem faktycznych fill’i.

Dane wejściowe: 50,184 przyczynowych snapshotów T−59; metadane Gamma dla 79,680 potwierdzonych rynków; czas wykonania 66.5 s. Model nie odtworzył zdarzeń ani nie trenował modelu.

- `historical_fee_regimes_v1.json` — kopia rejestru użytego w tym przeliczeniu.
- `gamma_fee_metadata_capture_all_markets.csv.gz` — metadane per `condition_id`, 79 680 rynków.
- `gamma_fee_regime_boundary_samples.csv` — rynki po obu stronach przejść taryfy i granice zakresu.
- `fee_rule_opportunity_audit.csv.gz` — wybrana reguła, parametry, jednostka, źródło i pewność dla wszystkich wejść T−59.
- `archive_fee_field_audit.csv` — rozkład i interpretacja pola archiwalnego.
- `scenario_summary.csv`, `monthly_by_scenario.csv`, `primary_trade_ledger.csv` — pełna siatka, miesięczne saldo/opłaty/transakcje/drawdown/pominięcia i szczegółowe transakcje.
- `counterfactual_comparison.csv` — etykiety oraz wyniki poprzedniego raportu obok nowej ścieżki.
- `independent_signal_diagnostic_ledger.csv.gz`, `independent_signal_diagnostic_summary.csv` — sygnały fixed-$5 po pierwszym ograniczeniu gotówką.
Luka historyczna obejmuje zapisane okazje T−59 od 2026-04-15 17:05 do 2026-10-06 23:55 UTC: Gamma `feeSchedule` pobrano 2026-10-10, a oryginalnych wartości z utworzenia rynku ani czasu matchu nie zachowano. Stawki i wykładniki są więc potwierdzone dla przechwyconych rekordów per rynek, lecz ich niezmienność w czasie pozostaje nierozstrzygnięta.
Etykieta `post_upgrade_confirmed_fee_segment` oznacza potwierdzoną jednostkę poboru i formułę V2; nie oznacza potwierdzonej historycznej stawki per rynek, bo jej wartość pochodzi z przechwycenia Gamma z 2026-10-10.
Dla V1 nadal brakuje faktycznej kwoty operatora i podziału maker-fill per match; dla V2 potwierdzona jest jednostka i udokumentowana formuła, lecz brak rzeczywistych opłat matchów. Dokładna sekunda przejścia V1/V2 w przybliżonym oknie 2026-04-28 11:00–12:00 UTC także pozostaje nierozstrzygnięta; wejścia z tego okna wyłączono.
