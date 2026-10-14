# Informacja BTC ponad rynek T−60 — 14 października 2026

## Werdykt

**Brak wykazanej informacji przyrostowej.** W istniejącym, przyczynowym porównaniu OOF model rynku rozszerzony o BTC uzyskał nieco lepsze średnie log-loss i Brier niż market-only, ale oba sparowane 95% przedziały blokowe obejmują zero. Przekroje według rozbieżności BTC–rynek nie są stabilne między foldami. Nie ma podstaw, by uznać tę rodzinę cech za stabilny sygnał ponad rynek.

To rozstrzygnięcie dotyczy informacji predykcyjnej; rentowność to odrębny wynik. Na developmentzie zamrożona polityka EV 1% przyniosła dodatni wynik na swojej stałej próbie zleceń. W historycznie eksponowanym holdoucie jej wykonane zlecenia straciły netto 732,16 USD mimo oczekiwanego zysku netto 1 242,33 USD według wyceny T−59. Holdout jest wyłącznie opisowy, nie jest niezależnym potwierdzeniem OOS i nie daje podstaw do gotowości live.

**Następny krok:** pozostawić live wyłączone i zebrać nowy, z góry zapisany okres obserwacyjny bez transakcji ani dostrajania; wrócić do testu dopiero po uzyskaniu wystarczającej liczby nowych bloków czasowych. Obecne wyniki nie uzasadniają dalszego strojenia tej rodziny cech.

## Zamknięcie poprzedniego eksperymentu

- Model update zachował `A_frozen`. `A_reoptimized` miał niższe developmentowe log-loss o 0,000338, lecz skorygowany przedział 97,5% wynosił [−0,000945; +0,000188], więc nie potwierdził poprawy według zamrożonej reguły wyboru.
- `B_dynamic_price_volume` zatrzymał się na 4/5 wymaganych udanych prób. Nie powstał model końcowy ani predykcje holdout; nie wznawiano B.
- Zakończona wcześniej diagnostyka prunera obejmowała 5 już przyciętych konfiguracji × 3 foldy development. Tylko jeden z 15 przebiegów pobił najlepszą ukończoną konfigurację tego samego foldu. Nie odtwarzała decyzji MedianPruner ani nie szacowała odsetka błędnych przycięć, więc nie daje rozstrzygającego dowodu szkodliwego przycinania.
- Żaden z tych wyników nie uzasadnia deploymentu, refitu deploymentowego ani zleceń. Live pozostało wyłączone.

Źródła: [raport model update](../btc_preopen_model_update_20261012_partial.md), [diagnostyka prunera](../b_pruner_development_diagnostic_20261014/pruner_diagnostic_report.md).

## Co już ustalono

Nazwy literowe są lokalne dla danego eksperymentu. W tabeli B z nested experiment oznacza `B_market_reference`; w model update B oznacza `B_dynamic_price_volume`.

| Pytanie | Istniejący eksperyment | Próba i okres | Wynik | Co pozostawało nierozstrzygnięte |
|---|---|---|---|---|
| Czy samo BTC przewyższa referencję rynku? | Nested probability: `A_BTC_calibrated` vs `B_market_reference` | 8 676 wspólnych OOF, maj–czerwiec 2026 | `B−A` ΔLL +0,000674 [−0,000127; +0,001474]; przedział obejmuje zero. W holdoucie rynek miał niższe LL, ale ten okres był wcześniej oglądany. | Czy BTC pomaga warunkowo przy podobnej cenie rynku i czy wynik utrzymuje się w kolejnych foldach. |
| Czy logistyczna kalibracja ceny rynku pomaga? | `C_market_logistic` vs `B_market_reference` | Te same 8 676 OOF | `C−B` ΔLL +0,000802 [+0,000176; +0,001535], gorzej dla C; ΔBrier +0,000404 [+0,000091; +0,000770]. | Czy BTC wnosi wartość do równie określonego modelu rynku. |
| Czy dodanie BTC do regresji rynku pomaga? | `D_market_plus_BTC` vs `C_market_logistic` | Te same 8 676 OOF | `D−C` ΔLL −0,000923 [−0,001887; +0,000101], ΔBrier −0,000462 [−0,000943; +0,000049], ΔAUC +0,006887 [−0,005807; +0,020980]. Punktowe wyniki są lepsze, ale przedziały LL i Brier obejmują zero. | Stabilność efektu według foldu oraz warunkowy rozkład UP przy zbliżonym `p_market`. |
| Czy aktywna korekta BTC nad rynkiem daje powtarzalny zysk jakości? | `E_probability_mixture` i `F_regularized_logit_correction` kontra market-only `B` i `D` | Te same 8 676 OOF | `E−B` ΔLL −0,000572 [−0,001243; +0,000125]; `E−D` −0,000451 [−0,001064; +0,000184]; `F−B` +0,000230 [−0,000111; +0,000602]. Żaden przedział LL nie wyklucza zera. | Czy wąski warunek cenowy lub inna podgrupa okaże się stabilna poza użytymi próbkami. |
| Czy wybrana polityka jest rentowna po kosztach? | Nested development selection i późniejsza ekonomika EV1; następnie osobny model update `A_frozen` | Nested OOF 8 676; w tym wcześniejszym raporcie 1 035 development filli EV1 i +264,52 USD. Model update używał innego wspólnego developmentu i tej samej reguły `A_frozen`. | Ekonomika jest wrażliwa na okres i definicję ledgeru. Audyt tutaj odtwarza stałą politykę `A_frozen` na jej własnym ledgerze: 1 133 filli development, +230,39 USD; holdout −732,16 USD. | Czy kalibracja i przewaga utrzymają się w nowym, wcześniej niezbadanym okresie. |

Pełne źródłowe tabele i procedura nested selection są w [raporcie nested probability](../btc_market_probability_nested_20261012.md). Starszy eksperyment marzec–maj używał innej definicji etykiety i nie miał prawidłowego snapshotu księgi przed startem, więc nie jest bezpośrednio porównywalny z tym testem T−60.

## Zamrożony protokół i dane

Protokół oraz source identity zapisano w [manifeście analizy](analysis_manifest.json) przed wygenerowaniem nowych przekrojów. Skrypt zatrzymuje się, jeżeli którykolwiek hash źródła nie zgadza się z manifestem. Użyto zapisanych predykcji `A_frozen`, oficjalnego indeksu rynków, istniejących snapshotów T−60/T−59, istniejącego ledgera decyzji i zapisanego nested OOF C/D. Nie trenowano ani nie refitowano BTC, nie odtwarzano archiwum księgi i nie uruchamiano diagnostyki prunera.

- Development: `market_start_utc < 2026-07-01T00:00:00Z`. Historyczny holdout obejmuje starty od 1 lipca do cutoffu danych `2026-10-07T00:00:00Z`; prezentuje się go wyłącznie opisowo.
- Jednostką jakości prognoz jest jeden rynek i jeden oficjalny wynik UP/DOWN. Predykcja BTC musi mieć przyczynowy znacznik dostępności zgodny z kontraktem: nominalna decyzja T−60 oraz łączny sekundowy budżet na predykcję, zlecenie i wykonanie do T−59. Przekroczenie terminu jest odrzucane.
- W próbie A nie użyto ceny ani wykonania T−59 do kwalifikacji. T−59 służy tylko do oceny wykonania już wybranych decyzji próby B.
- Manifest zapisuje różnicę hashy kontraktu: artefakt model update zachowuje wcześniejszy hash `57d5f59…`, a bieżący `docs/BTC_RESEARCH_CONTRACT.md` ma hash `4de218f…` po doprecyzowaniach w późniejszym commicie. Analiza przestrzega aktualnej treści kontraktu; nie zmienia historycznego freeze.

### `p_market` i cena zakupu

Zgodnie z poprzednim eksperymentem:

`m_up = (up_bid + up_ask) / 2`, `m_down = (down_bid + down_ask) / 2`, `p_market = m_up / (m_up + m_down)`.

To znormalizowana prognoza proxy z midpointów obu stron, a nie cena wykonania ani gwarantowane, skalibrowane prawdopodobieństwo. Do próby wymagane są prawidłowe bid i ask obu stron T−60 oraz istniejące kontrole pokrycia źródła, przyszłych zdarzeń i luk księgi. Mediana kwotowań w eksperymencie źródłowym wynosiła dla UP 0,500/0,510 bid/ask, a dla DOWN 0,490/0,500; mediana spreadu każdej strony to 0,01. Potencjalny zakup wycenia ask/VWAP i opłaty z zaakceptowanego kontraktu, a wykonanie T−59 używa istniejącego ledgera ask/VWAP, głębokości i opłat. Ask nie jest używany jako `p_market`.

Koszyki `p_market` i `p_BTC−p_market` mają po cztery poziomy. Granice foldów walidacyjnych wyznaczono wyłącznie z wcześniejszych foldów; jeśli kwantyle nie dawały czterech granic, obowiązywały zamrożone granice awaryjne `p_market=[0,48; 0,50; 0,52]` i różnicy `[−0,05; 0; 0,05]`. Dla holdoutu granice obu koszyków pochodzą z całego developmentu. Koszyki ceny T−60 dla polityki pochodzą wyłącznie z wcześniejszych sygnałów; holdout używa granic wyznaczonych na całym developmentzie. Komórki z `n<200` są nierozstrzygające i nie były łączone po wyniku. Niepewność częstości UP i kalibracji to 95% przedziały z ruchomych 3-dniowych bloków UTC, 1 000 replikacji, seed `20261002`. Głównym testem dodatkowej informacji pozostaje sparowane OOF C/D z istniejącym bootstrapem 3-dniowych bloków UTC, 1 000 replikacji, seed `20261012`.

## Próby i wykluczenia

| Okres / fold | Causal prediction rows | Próba A: wspólna cena | Próba B: sygnały EV1 | Próba C: fille T−59 | B bez fillu |
|---|---:|---:|---:|---:|---:|
| Development `dev_2026q2_a` | 4 403 | 2 119 | 66 | 64 | 2 |
| Development `dev_2026q2_b` | 8 928 | 5 253 | 463 | 456 | 7 |
| Development `dev_2026q2_c` | 8 640 | 3 423 | 617 | 613 | 4 |
| **Development razem** | **21 971** | **10 795** | **1 146** | **1 133** | **13** |
| Holdout historyczny, opisowy | 28 224 | 24 736 | 4 272 | 4 018 | 254 |

Próba A razem ma 35 531 rynków: 10 795 development i 24 736 holdout. Dodatkowe wiersze prediction-only w `calibration_seed` (30 157) i bez identyfikatora OOF (3 979) nie weszły do próby A; ich przyczyny wykluczeń są zapisane w CSV. Jednostka nie jest podwajana: asercje wymagają unikalnego `market_start_utc` i `condition_id` w każdej próbie, C musi być podzbiorem B, a outcome UP/DOWN jest liczony raz na rynek.

| Fold | Wykluczenia z A |
|---|---|
| `dev_2026q2_a` | 2 282 brakujące lub nieprawidłowe kwotowania obu stron; 2 bez potwierdzonego rynku Gamma. |
| `dev_2026q2_b` | 3 674 brakujące lub nieprawidłowe kwotowania; 1 bez Gamma. |
| `dev_2026q2_c` | 5 166 brakujących lub nieprawidłowych kwotowań; 36 niedostępnego pokrycia archiwum; 9 nieznanego stanu luki lub luki po inicjalizacji; 6 bez Gamma. |
| Holdout | 3 481 brakujących lub nieprawidłowych kwotowań; 4 przyszłe zdarzenia T−60; 2 bez Gamma; 1 bez wyniku oficjalnego. |

Dalsze powody nieprzejścia filtra B na T−60: w foldach development kolejno 2 734/4 409/5 490 niepriceable, 1 522/3 770/2 219 bez dodatniego EV, 79/285/308 poniżej zamrożonego progu 1%; holdout 5 109/16 865/1 976. Dwa/7/cztery sygnały development i 254 holdoutowe sygnały nie zostały wykonane T−59. Pełne liczniki są w [lejku prób](sample_funnel_by_fold.csv), [wykluczeniach A](sample_A_exclusions_by_fold.csv) i [powodach decyzji](policy_funnel_reasons_by_fold.csv).

## Czy BTC porządkuje częstość UP przy podobnym rynku?

Wspólna próba development ma średnio `p_market=0,504522`, `p_BTC=0,503901`, średnią różnicę BTC–rynek −0,000621; mediana różnicy to −0,001046, a P10/P90 −0,024531/+0,023287. Korelacja `p_market` z `p_BTC` to 0,297. Cały rynek miał 49,782% wyników UP; średnia `p_BTC` była wyższa o 0,608 punktu procentowego, ale sama różnica średnich nie pokazuje wartości przyrostowej.

| Development fold | n | Wynik UP | Średnie `p_market` | Średnie `p_BTC` | Średnie BTC−rynek | BTC−rynek P10 / mediana / P90 |
|---|---:|---:|---:|---:|---:|---:|
| `dev_2026q2_a` | 2 119 | 52,15% | 50,486% | 50,407% | −0,079 pp | −2,172 / −0,085 / +1,846 pp |
| `dev_2026q2_b` | 5 253 | 50,28% | 50,526% | 50,345% | −0,181 pp | −2,434 / −0,227 / +2,051 pp |
| `dev_2026q2_c` | 3 423 | 47,56% | 50,318% | 50,449% | +0,131 pp | −2,624 / +0,082 / +2,937 pp |
| **Razem** | **10 795** | **49,78%** | **50,452%** | **50,390%** | **−0,062 pp** | **−2,453 / −0,105 / +2,329 pp** |

Poniżej pokazano jeden uprzednio zdefiniowany zakres ceny rynku `M3` i częstość UP w koszykach resztowej różnicy. W pierwszych dwóch foldach `M3=[0,50; 0,52)`; w trzecim granice z wcześniejszych danych dały węższy `M3=[0,505; 0,510)`. Granice `R1–R4` także są fold-local i tylko z wcześniejszych danych; tabela raportuje ich wynik, a nie łączy komórek. Wiersze holdoutu z tymi samymi, developmentowymi granicami są w pełnym CSV i nie służyły do wyboru koszyków.

| Fold / zakres rynku | Koszyk różnicy BTC−rynek | n | Wynik UP [95% block CI] |
|---|---|---:|---:|
| A, `[0,50; 0,52)` | R2 (ujemna, środkowa) | 1 036 | 50,00% [47,53%; 52,66%] |
| A, `[0,50; 0,52)` | R3 (dodatnia, środkowa) | 964 | 54,88% [52,94%; 57,41%] |
| A, `[0,50; 0,52)` | R1 / R4 | 0 / 2 | nierozstrzygające |
| B, `[0,50; 0,52)` | R1 / R2 / R3 / R4 | 1 145 / 1 103 / 993 / 970 | 48,56% [46,39%; 51,24%] / 51,77% [49,14%; 53,97%] / 48,24% [45,25%; 51,41%] / 52,37% [49,20%; 55,33%] |
| C, `[0,505; 0,510)` | R1 / R2 / R3 / R4 | 548 / 316 / 363 / 588 | 43,80% [41,52%; 47,95%] / 43,35% [40,00%; 49,03%] / 45,18% [41,43%; 49,03%] / 51,70% [46,90%; 58,85%] |

W foldzie A częstość UP rosła między dwoma liczebnymi środkowymi koszykami. Fold B jest niemonotoniczny. W C najwyższy koszyk ma wyższą częstość UP niż pozostałe, ale jego przedział jest szeroki i nakłada się na pozostałe. To nie jest stabilny wzorzec między foldami. Porównania koszykowe są opisowe: resztowe koszyki nadal zawierają różne ceny rynku. Pełne 64 komórki, liczebności, średnie obu prognoz, dokładne granice i statusy są w [tabeli koszyków](market_residual_buckets.csv); rozkłady ogółem są w [tabeli foldów](market_distribution_by_fold.csv).

### Główny test przyrostowej jakości — istniejące nested OOF

Nie dopasowywano ponownie C ani D. Zweryfikowano, że zachowany plik zawiera 8 676 unikalnych wspólnych rynków oraz niepuste predykcje obu modeli; hash listy rynków i status braku refitu są w [tożsamości porównania](comparison_reuse_identity.json). Sparowane różnice są liczone jako D−C; ujemne wartości LL/Brier wspierają D.

| Fold | n | C LL / D LL | D−C LL | C Brier / D Brier | D−C Brier |
|---|---:|---:|---:|---:|---:|
| `dev_2026q2_b` | 5 253 | 0,693804 / 0,692977 | −0,000827 | 0,250328 / 0,249914 | −0,000414 |
| `dev_2026q2_c` | 3 423 | 0,693769 / 0,692700 | −0,001069 | 0,250311 / 0,249775 | −0,000536 |
| **Sparowane razem** | **8 676** |  | **−0,000923 [−0,001887; +0,000101]** |  | **−0,000462 [−0,000943; +0,000049]** |

Oba foldy mają punktowo lepsze wyniki D, ale tylko dwa zewnętrzne foldy nie dowodzą stabilności; pooled 3-dniowe przedziały obu głównych metryk obejmują zero. AUC D−C to +0,006887 [−0,005807; +0,020980]. Zgodnie z poprzednim nested report to jest główny test informacji przyrostowej, nie wybór zwycięzcy z koszyka.

## Ekonomika zamrożonej polityki EV 1%

Próba B kwalifikuje sygnały wyłącznie według ledgeru `A_frozen` na T−60: dodatni net EV i udział net EV co najmniej 1%, stały zlecenie brutto 5 USD. Strona nie zmienia się po T−60. Próba C to subset z rzeczywistym fill T−59. Oczekiwane EV T−60 i T−59 w poniższym moście dotyczą tych samych wypełnionych wierszy, a wynik zrealizowany uwzględnia istniejący accounting i opłaty.

| Okres | Sygnały B | Fille C | Obrót filli | EV netto T−60, te fille | EV netto T−59, te fille | Zmiana T−60→T−59 | Zrealizowany PnL netto | EV T−59 minus PnL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Development | 1 146 | 1 133 | 5 665,00 USD | 283,83 USD | 286,34 USD | +2,50 USD | +230,39 USD | +55,94 USD |
| Holdout historyczny | 4 272 | 4 018 | 20 090,00 USD | 1 269,08 USD | 1 242,33 USD | −26,75 USD | −732,16 USD | +1 974,49 USD |

Rozjazd w holdoucie jest znacznie większy niż zmiana samej wyceny przy przejściu T−60→T−59. To wskazuje na lukę prognoza–wynik dla tych historycznych transakcji, lecz nie identyfikuje jej przyczyny.

| Etap / okres | n | Średnie przewidywane prawdopodobieństwo wybranej strony | Faktyczna częstość wygranej |
|---|---:|---:|---:|
| A: wszystkie wspólne rynki, development (`p_BTC` UP) | 10 795 | 50,390% | UP 49,782% |
| B: sygnały EV1, development | 1 146 | 52,257% | 52,182% |
| C: fille, development | 1 133 | 52,256% | 52,074% |
| A: wszystkie wspólne rynki, holdout opisowy (`p_BTC` UP) | 24 736 | 50,040% | UP 49,830% |
| B: sygnały EV1, holdout opisowy | 4 272 | 52,681% | 49,017% |
| C: fille, holdout opisowy | 4 018 | 52,641% | 48,556% |

Na holdoucie odchylenie jest widoczne w obu stronach: 2 268 sygnałów DOWN / 2 196 filli miało oczekiwane EV netto T−59 +663,11 USD wobec zrealizowanego −445,99 USD; 2 004 sygnały UP / 1 822 fille miały odpowiednio +579,22 USD i −286,17 USD. Nie odwracano sygnału ani nie zmieniano progu.

### Opisowe przekroje holdoutu

Granice ceny T−60 są uprzednio ustalone z developmentu: P1–P4 mają krawędzie [0,48; 0,50; 0,51]. Granice resztowego BTC−rynek także pochodzą z developmentu. Poniższe wiersze nie zmieniają polityki.

| Przekrój holdoutu | Sygnały / fille | Obrót | EV netto T−59 | PnL zrealizowany netto | Hit rate filli |
|---|---:|---:|---:|---:|---:|
| T−60 cena P1 | 1 016 / 982 | 4 910 USD | +853,83 USD | −399,88 USD | 40,53% |
| T−60 cena P2 | 526 / 498 | 2 490 USD | +76,96 USD | −50,53 USD | 49,40% |
| T−60 cena P3 | 937 / 892 | 4 460 USD | +121,66 USD | +50,45 USD | 52,35% |
| T−60 cena P4 | 1 793 / 1 646 | 8 230 USD | +189,88 USD | −332,20 USD | 51,03% |
| BTC−rynek R1, najbardziej ujemna | 2 268 / 2 196 | 10 980 USD | +663,11 USD | −445,99 USD | 48,45% |
| BTC−rynek R4, najbardziej dodatnia | 2 004 / 1 822 | 9 110 USD | +579,22 USD | −286,17 USD | 48,68% |

Luka EV nie ogranicza się do wysokiego dodatniego `p_BTC−p_market`: stratne są oba skrajne koszyki R1 i R4, choć nie jest to dowód przyczynowy. Miesięcznie lipiec i październik miały ujemny PnL (−665,67 i −191,10 USD), a sierpień i wrzesień dodatni (+97,69 i +26,92 USD). To niestabilność okresowa, nie podstawa do filtrowania ani strojenia.

Wszystkie fille obu okresów korzystały z pełnej głębokości księgi; nie było filli proxy top-of-book. Tabele po stronach, cenie, reszcie, miesiącu i głębokości są w [ledgerze grup ekonomicznych](ev1_expected_vs_realized_groups.csv); granice cen są w [pliku granic](frozen_policy_price_bin_edges.csv), a etapy kalibracji w [tabeli kalibracji](policy_calibration_by_stage.csv).

## Odtworzenie i zakres

- Analiza przekrojów: `python analyze_btc_market_information_value_20261014.py` (bez argumentów CLI; używa zamrożonego manifestu i hash-zweryfikowanych lokalnych artefaktów).
- Czas całej analizy z weryfikacją wejść i zapisem tabel: **9,347 s**. Nie obejmuje wcześniejszego trenowania ani replayu, których w tym zadaniu nie uruchomiono.
- Testy celowane: `python -m unittest tests.test_btc_market_information_value -v` — 5/5 zaliczonych. Obejmują przyczynowość czasu predykcji, kwalifikację niezależną od pól T−59, unikalność rynków oraz podzbiór sygnałów/filli i niezależność wyboru od wyniku.
- Rezultat, hashe tabel i potwierdzenie braku treningu, replayu, selekcji holdoutu oraz zleceń zapisano w [results manifest](results_manifest.json).
- Live, deployment refit i zlecenia pozostają wyłączone.
