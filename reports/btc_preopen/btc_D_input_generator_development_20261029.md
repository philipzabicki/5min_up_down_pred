# Generator wejść D E0041 - tylko development - 2026-10-29

1. **Czy generator odtwarza dokładne wejścia E0041 z raw, w tym RP/VP i sześć taker_Z/V?** **Nie w całości.** Sześć kolumn taker_Z/V odtworzono z lokalnych archiwów USD-M i porównano z produktem development E0041. Dla memory/ESN odtworzono wynikowe PCA ze strumienia raw. BASE64 raw nie zostało odtworzone: w zamrożonym wejściu E0041 brakuje czterech dokładnych nazw kolumn, a aktywne RP/VP mają inne parametry.
2. **Czy stan jest zgodny, a świeży proces wznawia pracę poprawnie?** Memory/ESN PCA porównano z artefaktem dynamicznym E0041. Checkpoint końcowy zawiera stany rekurencyjne i bufory. Testy porównują nieprzerwany przebieg z kontynuacją po wznowieniu w świeżym procesie. Weryfikacja ukończonego przebiegu również zakończyła się w nowym procesie; dynamicznych wierszy nie przetworzono ponownie. Predykcje D są **BLOCKED**, bo cztery wejścia BASE64 nie są dostępne pod zamrożonymi nazwami.
3. **Jaki okres i ile wierszy faktycznie przetworzono; czy zapisano zweryfikowany stan końcowy development?** Raw: `[2020-06-09T09:33:00Z,2026-06-30T23:58:00Z)`, `3,186,145` wierszy, ostatni `Opened=2026-06-30T23:57:00Z`. Zapisano atomowy checkpoint końcowy. Modelowe wejścia kończą się na `Opened=2026-06-30T23:53:00Z`.
4. **Jakie były czas, pamięć, wolne miejsce i wyniki zgodności?** Pełny replay: `310.1s`, peak RSS `2.23 GiB`, wolny dysk przed/po `206.90/206.59 GiB`. Walidacja świeżego procesu: `133.796` s, peak RSS `1.08 GiB`, ponownie przetworzone dynamiczne wiersze `0`. Mismatche masek/liczb wierszy: `0`; statusy CSV: `BLOCKED_BASE64_MISSING=1, PASS=38, UNVERIFIED=1` (`2` rekordy mają status inny niż PASS).
5. **Co pozostaje zablokowane i jaki konkretny etap może nastąpić bez refitu?** BASE64 raw parity i predykcje D. W wejściowym Parquet brakuje dokładnie tych czterech zamrożonych kolumn: `KeltnerChannel_fit_5m_preopen_v1_pop64_atrmatypfba_atrper1151_maper533_matypnwmaepan_souhlcc4, ADX_fit_5m_preopen_v1_pop64_adxper452_atrper1990_matypadxkama_matypatrnwmatria_matypnegdmehma_matypposdmema_negdmper706_posdmper1716, StochOsc_fit_5m_preopen_v1_pop64_faskper24_slodmatyptrima_slodper1052_slokmatypnwmabiw_slokper1892, BollingerBands_fit_5m_preopen_v1_pop64_maper328_masouhlcc4_matypshmma_stdmaper53_stdmatypkama_stdsouhlcc4`. Kolejny developmentowy etap może odtworzyć te kolumny z historycznych konfiguracji indicator-fit i zamrożonego stanu RP/VP E0041 po wskazaniu dokładnych artefaktów oraz implementacji przetwarzania porcjami.

## Statusy bloków

| Blok | Status | Dowód / zakres |
|---|---|---|
| BASE64 / RP / VP | `BLOCKED_BASE64_ARTIFACTS_MISSING` | Cztery z 64 zamrożonych nazw BASE nie występują w schematach referencyjnych E0041; bieżące RP/VP różnią się od zamrożonych ustawień. |
| Flow 6 | `VERIFIED` | `3,198,234` wierszy referencyjnych, `3,198,238` odtworzonych; koniec odtworzenia `2026-06-30T23:57:00+00:00`. |
| Memory / ESN | `PARITY_PASS_VS_E0041` | Pełne odtworzenie cech dynamicznych z raw i porównanie z E0041; wiersze z luką lub nieprawidłową świecą są jawnie raportowane. |
| Predykcje 9 x D + Platt | `BLOCKED_BASE64_MISSING` | Predykcji nie wygenerowano, bo dokładne wejścia BASE nie są dostępne. Kalibratora nie dopasowywano. |
| Checkpoint | `VERIFIED_END_OF_DEVELOPMENT` | `data/analysis/polymarket/BTC/btc_D_input_generator_development_20261029/checkpoint_state.json`; SHA-256 `54e64361f24b46ca41a7eb33630344b0e7343c4b6223e234c21a18d8bcfcad1c`; zawiera memory, 3 ESN, raw tail 720 i flow tail do 10 095 minut. |

## Zgodność numeryczna

Plik [CSV zgodności](btc_D_input_generator_parity_20261029.csv) raportuje maski, maksymalny błąd bezwzględny i względny oraz pierwszy niezgodny czas dla każdej cechy. Zamrożone tolerancje E0044 pozostały bez zmian: flow/raw float64 `1e-10`, dynamic float32 `2e-6`; identyfikatory, czasy i maski porównano dokładnie. Porównanie nie służyło rankingowi, metrykom ani obliczeniom ekonomicznym.

## Odtwarzalność i produkty

- Uruchomienie i wznowienie: `python run_btc_D_input_generator_development_20261029.py`; runner automatycznie sprawdza tożsamość i integralność checkpointu.
- Feature contract: [`btc_D_input_generator_feature_contract_20261029.json`](btc_D_input_generator_feature_contract_20261029.json).
- Parity: [`btc_D_input_generator_parity_20261029.csv`](btc_D_input_generator_parity_20261029.csv).
- Manifest: [`btc_D_input_generator_manifest_20261029.json`](btc_D_input_generator_manifest_20261029.json); local runtime copy: `data/analysis/polymarket/BTC/btc_D_input_generator_development_20261029/run_manifest.json`.
- Checkpoint: `data/analysis/polymarket/BTC/btc_D_input_generator_development_20261029/checkpoint_state.json`.
- Panel predykcji: `data/analysis/polymarket/BTC/btc_D_input_generator_development_20261029/development_prediction_panel.json`; zawiera `0` wierszy. BASE64 było niekompletne, więc predykcji nie zapisano.
- Pełny replay: `310.1s`; peak RSS `2.23 GiB`; checkpoint `2,130,680` B; runtime output po pełnym replayu `337,501,245` B.
- Walidacja świeżego procesu: `133.796` s; peak RSS `1.08 GiB`; dynamiczne wiersze przetworzone ponownie: `0`.

## Przyszły etap holdout - tylko plan

Planowana kontrola źródeł zachowuje pełny kalendarz `28,224` slotów, `28,222` lokalnie potwierdzonych rynków i 2 nierozstrzygnięte sloty. `27,107` rekordów z pełnymi metadanymi archiwum to kandydaci do kontroli źródła, a nie obserwacje ewaluacyjne. `1,115` okien niekompletnych zachowuje podział: 816 bez entry-hour/freshness i 299 bez wcześniejszej historii.

Gotowość wejść modelu, księgi T-60/T-59, jakość źródła i rozliczenia należy raportować osobno. Brak księgi nie oznacza braku danych BTC ani resetu pamięci. Kwalifikację należy zamrozić na podstawie dostępności i jakości, niezależnie od predykcji; wykluczenie danych nie jest `no_trade`. Obowiązują E0038, D0019 oraz FAILED E0037 / BLOCKED T0009; development parity nie zamyka tych gate'ów. Nie utworzono `evaluation_rows.parquet` i nie wykonano ewaluacji.

## Granice tego przebiegu

Bundle E0041 pozostał niezmieniony: `D_BUNDLE_READY`; `evaluation_authorized=false`; `activated=false`; `HOLDOUT_NOT_RUN`. Nie odczytano nowych wartości holdoutowych, predykcji, etykiet, payoutów, ksiąg ani wyników ekonomicznych. Nie wykonano treningu, selekcji, optymalizacji, ponownego dopasowania kalibratora ani zdalnego pozyskania danych.
