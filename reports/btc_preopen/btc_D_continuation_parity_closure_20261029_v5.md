# E0047 - Zamkniecie zgodnosci kontynuacji generatora D

## Wyniki najwazniejszych kontroli

- **FLOW po historycznym cast:** 0 bledow reprezentacji na 3,186,141 kluczach x 6 cech; status PASS. Liczba **19 063 239** z E0046 byla blednie przypisana: agregat sumowal blok diagnostyczny float64 przed castem. Hash-sprawdzenie CSV E0046 wykazalo 0 bledow wylacznie w rekordach `flow_representation_float32`; E0047 agreguje tylko ten blok i wymaga pelnego, rozlacznego pokrycia kluczy.
- **BollingerBands:** E0046 roznil sie od batch na wszystkich 38 977 / 1 440 / 40 417 wierszach odpowiednich sufiksow, bez roznic masek; pierwszy blad `2026-06-02T22:21:00Z`, maks. abs `0.00012635984263198452`. Obciecie wejscia resetowalo rekurencyjny akumulator LMA(8), a ogon nie zachowywal jego ciaglego stanu. E0047 przenosi dokladne akumulatory LMA i 593 poprzednie `diff2` dla zamrozonego NWMA_EPAN(594). Wynik: PASS.
- **ChaikinOsc:** E0046 roznil sie na wszystkich 38 977 / 1 440 / 40 417 wierszach, bez roznic masek; pierwszy blad `2026-06-02T22:21:00Z`, maks. abs `0.006234453991055489`. ADL i jego przyrosty byly bitowo zgodne z TA-Lib; roznice wprowadzalo ponowne zainicjowanie WMA w HMA(893) na obcietym ogonie. E0047 przenosi stan trzech akumulatorow TA-Lib WMA oraz okna wejsciowe; NWMA_EPAN(8) zachowuje siedem poprzednich wartosci. Wynik: PASS.
- **RP/VP:** koncowe stany E0047 porownano dokladnie z E0041 oraz miedzy niezaleznymi wznowieniami. RP: PASS; roznice reprezentacyjne: E0041 `$.feature_columns, $.horizon_names`, sciezki wznowien `brak roznic`. VP: PASS; roznice reprezentacyjne: E0041 `$.feature_columns, $.horizon_names`, sciezki wznowien `brak roznic`. W E0046 RP `horizon_names` i `feature_columns` mialy rownowazne tuple/list, a `last_candle_time` zapisywal ten sam UTC instant z/bez `+00:00`; te waskie roznice nie zmieniaja stanu. W E0046 VP `raw_profile` roznil sie numerycznie (18/11/15 elementow dla short/medium/long; maks. abs `7.275957614183426e-12`, `7.450580596923828e-09`, `3.725290298461914e-09`). Przyczyna byla inna kolejnosc dzialan kontynuacji wzgledem zamrozonego batch; E0047 odtwarza kolejnosc batch i nie normalizuje tablic liczbowych.
- **Macierze/predykcje po wznowieniu:** macierze PASS, predykcje techniczne PASS. Kazdy wiersz wszystkich trzech rzeczywistych sufiksow przeszedl przez wznowione BASE64/FLOW6/memory/ESN, historyczne konwersje i trzy macierze po 86 cech. Przeciecia zamrozonego panelu i hashe kazdego zrodla sa w manifeście/CSV. Panel finalizacyjny: C1->END 64/64, C2->END 64/64.
- **Checkpointy:** oczekiwana kompletnosc C1, C2 i END oraz zgodnosc C1->C2->END z niezaleznym C1->END: PASS. Zakres raw: 3 186 145 wierszy, `Opened < 2026-06-30T23:58:00Z`; ostatni raw `23:57Z`, ostatni modelowy `23:53Z`.

## Status blokow

| Blok | Wynik |
|---|---|
| FLOW_formula | PASS |
| FLOW_representation | PASS |
| Bollinger_continuation | PASS |
| Chaikin_continuation | PASS |
| BASE64_continuation | PASS |
| FLOW_continuation | PASS |
| memory_ESN_continuation | PASS |
| batch_model_matrices | PASS |
| batch_D_predictions | PASS |
| resumed_model_matrices | PASS |
| resumed_D_predictions | PASS |
| final_states | PASS |
| all_parity_records | PASS |

V3 zwrocilo etykiete `DEV_GENERATOR_VERIFIED`, ale nie objelo ona dwoch porownan semantycznych checkpointow. Porownywarka nie obslugiwala ogonow DataFrame, a agregacja statusu pominela wiersze tych kontroli. Etykieta i status v3 sa niewazne. Proba v4 zatrzymala sie przed publikacja wyniku z powodu braku katalogu `suffixes`; nie przypisuje sie jej statusow. V5 tworzy wymagany katalog, ponownie sprawdza hash-pinned produkty, porownuje checkpointy kolumna po kolumnie i blokuje status VERIFIED, jesli dowolny rekord parity ma status FAIL.

## Przebiegi i zasoby

Zakresy wyprodukowane w v3: C1->C2 38,977 wierszy (2026-06-02T22:21:00..2026-06-29T23:57:00); C1->END 40,417 (2026-06-02T22:21:00..2026-06-30T23:57:00); C2->END 1,440 (2026-06-29T23:58:00..2026-06-30T23:57:00). Produkty sufiksow powstaly w swiezym przebiegu v3; v5 weryfikuje je z hash-pinned plikow i nie parsuje ani nie przelicza wierszy surowego prefiksu/sufiksow. Kontrola SHA-256 czyta zapisane bajty plikow. Pelny przebieg v3 trwal 354.9 s, peak RSS 1.73 GiB; biezacy audyt porownawczy trwal 176.9 s.  Szczyt RSS audytu i zrodlowego przebiegu 1.73 GiB. Dostepny RAM przed/po: 46.06/44.55 GiB; wolny dysk: 203.64/203.60 GiB. W audycie v5 sparsowano 0 wierszy surowego prefiksu i przeliczono 0 surowych wierszy sufiksu; kontrole hash czytaja bajty plikow. Szczegoly I/O v3 i bajtow hash-zweryfikowanych w v5 sa w raporcie JSON.

Nie wykonano treningu, selekcji, fitow, metryk jakosci, analizy ekonomicznej ani evaluatora. Nie odczytano wartosci ani artefaktow holdoutu; nie wygenerowano nowych predykcji holdoutu. Pozostaja `evaluation_authorized=false`, `activated=false`, `HOLDOUT_NOT_RUN`; hash finalnego bundle E0041: `7715419ce5efbce8405db861fb3a67c481038c07f11eec019137f14528f0f97d`. E0046 i jego poczatkowe/korygowane artefakty zachowano bez zmian. E0037 pozostaje `FAILED`, T0009 `BLOCKED`, a T0046/T0048 pozostaja nietkniete.

`DEV_GENERATOR_VERIFIED` oznacza wylacznie status developmentowego generatora. Kolejnym bezpiecznym etapem jest przygotowanie osobnego, metadata-only manifestu pokrycia historycznych zrodel Binance index/OHLCV i USD-M FLOW dla przyszlego wejscia; wymaga ono zachowania obecnych bramek i nie obejmuje payloadu holdoutu.

Artefakty: [btc_D_continuation_parity_closure_20261029_v5_manifest.json](btc_D_continuation_parity_closure_20261029_v5_manifest.json), [btc_D_continuation_parity_closure_20261029_v5_parity.csv](btc_D_continuation_parity_closure_20261029_v5_parity.csv), [btc_D_continuation_parity_closure_20261029_v5_feature_contract.json](btc_D_continuation_parity_closure_20261029_v5_feature_contract.json).
