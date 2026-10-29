# E0046 - FLOW representation and stateful BASE64 continuation

1. Czy roznice FLOW w calosci wyjasnia historyczna konwersja? **Nie**. Surowe FLOW6 obliczono jako `float64`. E0041 wykonuje dokladny join `Opened` w [`run_btc_preopen_experiment.py`](../../run_btc_preopen_experiment.py), `_join_causal_candidate_features` (linie 1311-1375): wyznacza indeksy, dopasowuje wiersze, rzutuje dopasowane wartosci na `float32`, pozostawia niedopasowania jako NaN i zapisuje kolumny Arrow `float32`. Join nie imputuje ani nie transformuje FLOW. Macierze sa laczone w `_family_matrix`; finalizer rzutuje wejscie na `float32` przed LightGBM. Po wiernym cast/join pozostalo **19063239** niezgodnosci w 3,186,141 wspolnych wierszach x 6 cech. Wczesniejsze porownanie float64 do produktu zapisalo 19,063,239 roznych wartosci; max abs 6.69576539e-05, max rel 5.95939645e-08. E0045 zachowuje dowod formuly float64: 6/6 cech, 3,198,234 wierszy, tolerancja 1e-10/1e-10, max abs 0. Cztery wiersze po ostatniej referencji FLOW (2026-06-30T23:54:00+00:00 - 2026-06-30T23:57:00+00:00) sa osobno oznaczone.

2. Czy wszystkie macierze wejsciowe finalnego D sa zgodne na calym dozwolonym wspolnym zakresie? **Tak**. Porownano 3 macierze x 86 uporzadkowanych cech x 3,186,141 wierszy, po historycznym cast `float32`, na kluczach `Opened` od 2020-06-09T09:33:00+00:00 do 2026-06-30T23:53:00+00:00. Nie uzyto etykiet.

3. Czy kazdy z 64 BASE komponentow wznowil prace w swiezym procesie i czy oba rzeczywiste sufiksy oraz stany koncowe sa zgodne? **Nie**. Wszystkie 64 obliczono na kazdym sufiksie, lecz tylko 62/64 przeszly parity (BollingerBands i ChaikinOsc nie przeszly). C1: 40,417 wierszy, 2026-06-02T22:21:00 - 2026-06-30T23:57:00; C2: 1,440 wierszy, 2026-06-29T23:58:00 - 2026-06-30T23:57:00. Kazdy checkpoint zapisano atomowo, a kazde wznowienie dzialalo w osobnym interpreterze. Niezalezne sciezki C1 i C2 zgadzaja sie miedzy soba, lecz scisle porownanie koncowych stanow RP/VP z zapisanym stanem E0041 nie przeszlo.

4. Czy dziewiec modeli, agregacja i Platt zachowaly zgodnosc na panelu? **Tak**. Sprawdzono 154 wierszy panelu E0045, w tym 31 minut decyzyjnych i 64 minut finalizacyjnych 2026-06-30 22:50-23:53; bez etykiet i metryk jakosci.

5. Jaki jest czas, peak RAM, I/O, ilosc ponownie przeliczonych danych i rozmiar checkpointu? Calkowity czas 327.4 s; maksymalny peak RSS parent/worker 1.70 GiB. Prefiks: 3,145,728 raw wierszy i 306,798,503 bajtow; nowe wiersze suffixow przetworzone we wszystkich wznowieniach: 80,834. Builder BASE64 ponownie obliczyl 48,004 wierszy ogona, a FLOW6 40,380; wiersze wejsciowe buildera BASE64 lacznie: 128,838; raw bajty odczytane przy resume: 7,746,736. Hashowanie archiwow i odczyt ich strumieni FLOW: 35,616,799 bajtow. Rozmiary checkpointow i czasy budowy, odczytu stanu, etapow wznowienia oraz zapisu sa zapisane w manifeście. Projekcje Parquet stanowia osobno raportowana dolna granice I/O.

## Statusy

| Blok | Wynik |
|---|---|
| FLOW_formula | PASS |
| FLOW_representation | FAIL |
| model_matrices | PASS |
| BASE64_continuation | FAIL |
| FLOW_continuation | PASS |
| memory_ESN_continuation | PASS |
| final_D_predictions | PASS |
| final_states | FAIL |

## Korekta statusu macierzy

Pierwsza wersja raportu blednie oznaczyla `model_matrices` jako `FAIL`, poniewaz agregator oczekiwal jednego rekordu na ceche i seed. Porownania sa zapisywane w 49 blokach danych. Audyt parity CSV potwierdzil dla kazdego z trzech seedow po 86 cech, 49 rekordow na ceche, sume dokladnie 3,186,141 wierszy na ceche oraz status `PASS` dla kazdego rekordu (12,642 rekordy lacznie). Status komponentu skorygowano na `PASS`; nie zmieniono danych, porownan, kluczy ani tolerancji. Oryginalne pliki zachowano jako [v3 report](btc_D_flow_representation_and_base64_continuation_20261029_v3_initial.md) i [v3 manifest](btc_D_flow_representation_and_base64_continuation_20261029_v3_initial_manifest.json). Pozostale statusy i ogolny `DEV_GENERATOR_PARTIAL` nie ulegly zmianie.

## Zakres i reprodukcja

E0046 uzywa wylacznie bezpiecznego raw development do `Opened=2026-06-30T23:57Z`; nie odczytano payloadu `23:58Z`. Checkpointy serializuja streak, RP/VP, ADX TEMA, Chaikin ADL, memory i ESN; FLOW6 utrzymuje 10,095 raw wierszy. Niezalezne sciezki C1/C2 zgadzaja sie miedzy soba, ale BASE64 parity wynosi 62/64, a scisle koncowe stany RP/VP nie sa identyczne z referencja E0041. Konfiguracje, transformaty i dziewiec modeli E0041 pozostaly zamrozone. Nie wykonano refitow, selekcji, treningu, metryk jakosci ani ekonomiki.

`DEV_GENERATOR_PARTIAL` opisuje tylko generator developmentowy. Nie zamyka bramki zrodel holdoutu ani nie oznacza rentownosci. Jesli wszystkie bloki przejda, przyszly osobny etap wymaga surowych Binance index/ohlcv i USD-M FLOW, danych futures/index do basis, pelnych ksiazek ze znacznikami venue/receive i ciagloscia, a nastepnie projekcji BASE70+dynamic z zachowaniem warmup i stanowych transformow. E0043 pozostawia nierozwiazana luke 68 godzin `[2026-08-15T10:00Z,2026-08-18T06:00Z)` bez kwalifikujacych pelnych ksiazek, dwa niedopasowane lokalne starty (`2026-07-12T02:25Z`, `2026-09-03T15:20Z`) i 1,115 okien bez potwierdzonej kompletnej historii. Osobny guard evaluatora nadal musi bronic granicy danych i wymagac odrebnego upowaznienia. Historycznych wejsc holdoutu ani evaluatora tu nie uruchomiono.

Artefakty: [btc_D_flow_representation_and_base64_continuation_20261029_manifest.json](btc_D_flow_representation_and_base64_continuation_20261029_manifest.json), [btc_D_flow_representation_and_base64_continuation_20261029_parity.csv](btc_D_flow_representation_and_base64_continuation_20261029_parity.csv), [btc_D_flow_representation_and_base64_continuation_20261029_feature_contract.json](btc_D_flow_representation_and_base64_continuation_20261029_feature_contract.json).
