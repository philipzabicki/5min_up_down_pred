# Naprawa loadera finalnego D E0041 - development - 2026-10-29

## Wynik

E0044 sprawdzal hash finalnego manifestu E0041, ale specyfikacje cech i sciezki modeli pobieral ze starszego `D_manifest.json`. Laczenie tych dwoch bundle powodowalo falszywy raport o czterech brakujacych cechach. E0045 korzysta wylacznie z finalnego manifestu `7715419ce5efbce8405db861fb3a67c481038c07f11eec019137f14528f0f97d` i potwierdza, ze kopia finalizatora jest identyczna. Nazwy, kolejnosc, seedy, modele, sidecary, liczba drzew, transformy i Platt sa sprawdzane w tym samym bundle.

| Dawna nazwa loadera | Wymagana przez finalny E0041 | Cecha finalnego E0041 z tej samej rodziny |
|---|---:|---|
| KeltnerChannel_fit_5m_preopen_v1_pop64_atrmatypfba_atrper1151_maper533_matypnwmaepan_souhlcc4 | nie | KeltnerChannel_fit_5m_preopen_v1_pop64_atrmatypnwmagauss_atrper716_maper22_matypfba_souopen |
| ADX_fit_5m_preopen_v1_pop64_adxper452_atrper1990_matypadxkama_matypatrnwmatria_matypnegdmehma_matypposdmema_negdmper706_posdmper1716 | nie | ADX_fit_5m_preopen_v1_pop64_adxper640_atrper698_matypadxtema_matypatrwma_matypnegdmgma_matypposdmnwmacos_negdmper1756_posdmper1107 |
| StochOsc_fit_5m_preopen_v1_pop64_faskper24_slodmatyptrima_slodper1052_slokmatypnwmabiw_slokper1892 | nie | brak; finalny bundle nie zawiera tej cechy |
| BollingerBands_fit_5m_preopen_v1_pop64_maper328_masouhlcc4_matypshmma_stdmaper53_stdmatypkama_stdsouhlcc4 | nie | BollingerBands_fit_5m_preopen_v1_pop64_maper8_masouhlcc4_matyplma_stdmaper594_stdmatypnwmaepan_stdsouhlc3 |

Wlasciwy finalny bundle wymaga **0/4** dawnych nazw; nie zastosowano aliasow ani wypelniania zerami. Z surowego bezpiecznego prefiksu odtworzono wszystkie 64 BASE cechy dla 3,186,145 wierszy. Typy kolumn zostaly porownane dokladnie z kanonicznym produktem: `{'ADX_fit_5m_preopen_v1_pop64_adxper640_atrper698_matypadxtema_matypatrwma_matypnegdmgma_matypposdmnwmacos_negdmper1756_posdmper1107': 'double', 'BollingerBands_fit_5m_preopen_v1_pop64_maper8_masouhlcc4_matyplma_stdmaper594_stdmatypnwmaepan_stdsouhlc3': 'double', 'ChaikinOsc_fit_5m_preopen_v1_pop64_fasmatypnwmaepan_fasper8_slomatyphma_sloper893': 'double', 'KeltnerChannel_fit_5m_preopen_v1_pop64_atrmatypnwmagauss_atrper716_maper22_matypfba_souopen': 'double', 'candle_body_pressure_15m_lag1': 'double', 'candle_body_pressure_15m_lag2': 'double', 'candle_body_pressure_1d_lag2': 'double', 'candle_body_pressure_1h_lag1': 'double', 'candle_body_pressure_30m_lag1': 'double', 'candle_body_pressure_30m_lag5': 'double', 'candle_body_pressure_4h_lag1': 'double', 'candle_body_pressure_4h_lag7': 'double', 'candle_close_location_value_15m_lag1': 'double', 'candle_close_location_value_15m_lag2': 'double', 'candle_close_location_value_1h_lag1': 'double', 'candle_close_location_value_1h_lag5': 'double', 'candle_close_location_value_30m_lag1': 'double', 'candle_close_location_value_30m_lag3': 'double', 'candle_close_location_value_4h_lag1': 'double', 'candle_close_location_value_4h_lag7': 'double', 'candle_ret_co_5m_lag1': 'double', 'candle_ret_co_5m_lag2': 'double', 'candle_signed_vol_15m_lag1': 'double', 'candle_streak_15m': 'int32', 'candle_streak_30m': 'int32', 'candle_streak_3m': 'int32', 'candle_wick_asym_15m_lag1': 'double', 'candle_wick_asym_1d_lag1': 'double', 'candle_wick_asym_1d_lag7': 'double', 'candle_wick_asym_1h_lag1': 'double', 'candle_wick_asym_1h_lag2': 'double', 'candle_wick_asym_1h_lag6': 'double', 'candle_wick_asym_30m_lag1': 'double', 'candle_wick_asym_30m_lag3': 'double', 'candle_wick_asym_4h_lag1': 'double', 'futures_index_basis_change_4h': 'double', 'futures_index_basis_rel_15m': 'double', 'futures_index_basis_rel_1h': 'double', 'futures_index_basis_rel_30m': 'double', 'futures_index_basis_rel_4h': 'double', 'futures_index_basis_rel_5m': 'double', 'realized_volatility_15m': 'double', 'realized_volatility_1h': 'double', 'realized_volatility_compression_expansion_15m_1h': 'double', 'realized_volatility_down_15m': 'double', 'realized_volatility_down_1h': 'double', 'realized_volatility_down_5m': 'double', 'rp_short_imbalance': 'double', 'rp_short_resistance_above': 'double', 'rp_short_resistance_center_dist': 'double', 'rp_short_support_center_dist': 'double', 'session_close_impulse_B3': 'double', 'session_close_impulse_BVC': 'double', 'session_close_impulse_NYSE': 'double', 'session_close_impulse_Xetra': 'double', 'session_open_impulse_B3': 'double', 'session_open_impulse_NYSE': 'double', 'vp_all_local_above_below_volume_log_ratio': 'double', 'vp_long_local_above_below_volume_log_ratio': 'double', 'vp_medium_local_above_below_volume_log_ratio': 'double', 'vp_short_local_above_below_volume_log_ratio': 'double', 'vp_short_log_density_ratio_to_current_bin_minus_1': 'double', 'vp_short_log_density_ratio_to_current_bin_minus_3': 'double', 'vp_short_log_density_ratio_to_current_bin_plus_2': 'double'}`. Cechy RP/VP odtworzono w jednej pelnej partii, a zapisane stany koncowe zgadzaja sie z E0041. Kontynuacji stanowej BASE generatora nie zweryfikowano.

FLOW6 odtworzone z surowych archiwow przechodza porownanie z referencja FLOW dla 6/6 cech i 3,198,234 wierszy. Dodatkowe 4 wiersze na koncu przebudowanego FLOW pozostaja poza zakresem referencji. Osobne, zamrozone porownanie wartosci float64 z surowej odbudowy do finalnego BASE70 E0041 przy tolerancji `1e-10` nie przechodzi dla 6/6 kolumn FLOW: 18,806,878 porownan feature-row ma blad numeryczny; pierwszy blad jest w `Opened=2020-06-09T09:33:00+00:00`, maksymalny blad abs/rel to `6.69576539e-05` / `5.95939645e-08`. Kanoniczny BASE70 zapisuje te kolumny jako Arrow `float`, podczas gdy odbudowa ma dtype float64; tolerancji nie zmieniono, a status FLOW pozostaje niezaliczony.

Wszystkie 64 cechy BASE64 i wszystkie 32 cechy memory/ESN przechodza porownanie z E0041 dla 3,186,145 wierszy. Surowe i kanoniczne wejscia podano do tych samych dziewieciu modeli finalnego E0041; predykcje kazdego czlonka, srednia 1/9 i zamrozony Platt przechodza porownanie.

Dla panelu finalizacyjnego sprawdzono dokladnie 64 zadeklarowane minuty, od 2026-06-30 22:50 do 23:53 UTC, z dziewiecioma predykcjami na minute. Pelny panel obejmuje 154 wybranych minut, z czego 31 to minuty decyzji.

Nowy proces odtworzyl checkpoint po `Opened=2026-06-02T22:20:00+00:00` i przetworzyl niepusty sufiks 40,417 wierszy do `Opened=2026-06-30T23:57:00+00:00`. Checkpoint zawiera stan memory/ESN, 720-minutowy ogon raw, ogon zrodel FLOW, kursory oraz tozsamosci i hashe zrodel.

## Statusy E0045

| Blok | Status |
|---|---|
| Tozsamosc loadera | VERIFIED_FINAL_E0041 |
| BASE64 parity z raw | PASS_RAW_PARITY_BATCH_ONLY |
| FLOW6 | FAILED_OR_UNVERIFIED |
| Memory/ESN | PARITY_PASS_VS_E0041 |
| Dziewiec modeli, srednia i Platt | PASS_FINAL_E0041_RAW_VS_CANONICAL |
| Kontynuacja memory/ESN w nowym procesie | VERIFIED_NONEMPTY_SUFFIX |
| Kontynuacja stanowa BASE64 | NOT_VERIFIED_BATCH_ONLY |
| Pelny stanowy generator | NOT_VERIFIED_BATCH_ONLY |

Tabela CSV raportuje status kazdej cechy/modelu, maski NaN/Inf, bledy numeryczne, najwieksze bledy i pierwszy niezgodny czas. Liczba odnotowanych niezgodnosci masek i wartosci: 18806878. Tolerancje: cechy BASE/FLOW `1e-10`, dynamiczne `2e-6`, predykcje `2e-12`; timestampy, kolejnosc oraz maski sa porownywane dokladnie.

## Czas i zasoby

- Odtworzenie BASE64: 118.2 s.
- Odtworzenie / porownanie FLOW6: 64.7 / 163.0 s.
- Dynamiczny prefiks / wznowiony sufiks: 209.3 / 6.5 s.
- Porownanie cech / predykcje / I/O checkpointu: 172.9 / 3.045 / 6.927 s.
- Czas scienny: 751.7 s; szczytowy RSS: 11.49 GiB; wolne miejsce przed/po: 205.44 / 204.62 GiB.
- Lokalne artefakty runtime: 871,346,256 B. Parquet BASE64 i checkpoint pozostaja lokalne.

## Walidacja

- Testy development: 23 passed in 9.06s.
- Pelny zestaw: 802 passed, 6 warnings, 220 subtests passed in 133.81s.
- `py_compile`: passed; `git diff --check`: passed on staged final changes.

## Granice i zachowane statusy

- E0044 pozostaje historycznym `PARTIAL`; korekta jego identyfikacji jest zapisana w E0045 i kanonicznych rekordach.
- E0041 pozostaje bez zmian: `evaluation_authorized=false`, `activated=false`, `HOLDOUT_NOT_RUN`.
- Nie odczytywano wartosci, cech, stanow, predykcji, etykiet, payoutow ani ksiag holdoutu. Nie uruchomiono evaluatora, live/shadow, zlecen, treningu, tuningu, selekcji ani dopasowania transformow/kalibratora.
- Stanowa kontynuacja BASE64 i pelny generator pozostaja `NOT_VERIFIED`; techniczne wyniki development nie zamykaja bramek E0037/T0009 ani luki E0043.

## Artefakty

- Protokol: `configs/research/btc_D_input_generator_final_bundle_fix_20261029_v2.json`
- Kontrakt: `btc_D_input_generator_final_bundle_fix_20261029_feature_contract.json`
- Tabela parity: `btc_D_input_generator_final_bundle_fix_20261029_parity.csv`
- Manifest: `btc_D_input_generator_final_bundle_fix_20261029_manifest.json`
- Panel i checkpoint: `data/analysis/polymarket/BTC/btc_D_input_generator_final_bundle_fix_20261029_v2/development_prediction_panel.json`, `data/analysis/polymarket/BTC/btc_D_input_generator_final_bundle_fix_20261029_v2/checkpoint_state.json`
