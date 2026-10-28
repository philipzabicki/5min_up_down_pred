# Przygotowanie przyszłej oceny holdoutu D

Data: 2026-10-28. Eksperyment E0041, zadanie T0012, decyzja D0021. Raport opisuje przygotowanie do jednego, z góry ustalonego przebiegu; nie jest raportem z wyniku holdoutu.

## Status

- `EVALUATION_CODE_READY`: runner oceny i testy mechaniki są gotowe; `evaluation_authorized` pozostaje `false`.
- `D_BUNDLE_READY`: **nie**. V2 ukonczyl selekcje developmentowa i strojenie, lecz nie doszedl do finalnego fitu dziewiecioczlonowego D, OOF ani kalibratora. Pierwsze wykonanie przerwal `KeyError` w metadanych migracji po selekcji; brakujace pole odtworzono z v2. Biezace wznowienie jest zablokowane przez 80 GiB gate (73.84 GiB wolne). Bundle v2 nie powstal.
- Dane: oficjalne Gamma i kalendarz rozstrzygnięć są `DATA_PRESENT_BY_METADATA`; T-60/T-59 w aktualnym pakiecie są `DATA_BLOCKED`; pokrycie OHLCV/przepływów, jakość treści, fee i mapowanie do wszystkich slotów są `DATA_NOT_VERIFIED`.
- `HOLDOUT_NOT_RUN`: prawda. Nie odczytano holdout payloadu, nie wygenerowano predykcji, nie uruchomiono scorera ani ekonomiki.

Historyczna pierwsza proba v1: `blocked_by_resource_gate`; pozniejsza proba v1 przeszla prog i wyczerpala budzet RP. Pierwsze dwa wywolania v2 zatrzymaly sie przed odczytem danych. Pozniejsze wznowienie uzylo wylacznie bezpiecznego development prefixu; biezaca ponowna proba zatrzymala sie na niezmienionym 80 GiB gate.

Raport i tabela [pokrycia z metadanych](btc_D_holdout_coverage_metadata_20261028.csv) korzystają wyłącznie z istniejących protokołów, manifestów, ścieżek, hashy i wcześniej opublikowanych zakresów/liczności. Nie pobierano źródeł ani nie sprawdzano zawartości holdoutu.

## Zamrożony pipeline

Holdout to półotwarty zakres startów `[2026-07-01T00:00:00Z, 2026-10-07T00:00:00Z)`. Oczekiwane są 28 224 pięciominutowe sloty kalendarza; to arytmetyka czasu, nie liczba potwierdzonych rynków. Pierwsza decyzja wypada `2026-06-30T23:59:00Z`, dlatego końcowy fit przyjmuje wyłączne ograniczenie dostępności etykiet przed tą chwilą. Każdy etap uczenia dodatkowo wymaga `Opened < fit_cutoff` i `label_available_at < fit_cutoff`.

E0041 wybiera D na dozwolonym developmencie. E0039/E0040 developmentowe konto fixed-5 osiągnęło `$374.574785` z `$100` przy drawdownie kosztowym `64.656%`; to uzasadnienie developmentowe podane w instrukcji, nie metryka holdoutu. Stały okres i rodzina A/B/C/D były już wcześniej eksponowane, więc ten przebieg nie jest dziewiczym ani w pełni niezależnym testem.

Metoda została zamrożona przed fit:

- Jedna selekcja zgodna z ograniczoną procedurą E0039 v7, wyłącznie na etykietach dostępnych przed `2025-10-01T00:00:00Z`. Nie wybiera się wariantu F1/F2/F3 po PnL.
- D = BASE70 + memory PCA8 + ESN PCA8, te same zweryfikowane transformaty, kolejność cech, reset luk i pełna siatka 3 seedów LightGBM × 3 seedów ESN; średnia dziewięciu członków.
- Jeden końcowy Platt na przyczynowych D OOF z fit cutoff `2025-10-01T00:00:00Z`, prognozowanych dla `[2025-10-01, 2026-07-01)`, z oficjalnymi etykietami Gamma dostępnymi przed pierwszą decyzją holdoutu. Wymagane jest minimum 1 000 wierszy i obie klasy; brak fallbacku Binance.
- Przyszła ekonomika używa jednego ciągłego `$100` konta na każde połączenie kontroli i opóźnienia: fixed `$5` oraz 5% wolnej gotówki bez zaokrąglania do minimum, każde dla `[1,5,15]` sekund. T-60 ustala stronę i rozmiar po dodatnim EV; T-59 używa pełnego natywnego fillu i uwzględnia wypłaty dostępne do tej sekundy. Pozycje pozostają do oficjalnego rozstrzygnięcia Gamma, a ostatnie pozycje rozliczają się terminalnie bez nowych zakupów poza oknem.
- Dodatni wynik oznacza terminalne equity fixed-5 ściśle powyżej `$100` we wszystkich trzech opóźnieniach. Raport ma pokazać każde opóźnienie, obie kontrole, drawdown kosztowy, fill’e, opłaty, przepływy miesięczne, równoczesne pozycje i reżimy. Kryterium kwotowe nie dowodzi istotności statystycznej ani gotowości produkcyjnej.

Reguła kalendarza, zamrożona przed oceną: single price `[2026-07-01,2026-08-07)`, TWAP30 `[2026-08-07,2026-08-14)`, TWAP60 `[2026-08-14,2026-10-07)`. Etykietą wypłaty pozostaje Gamma `target_polymarket_up`, nie Binance proxy.

## Dane i jawne braki

Poprzedni opublikowany raport podaje 28 220 Gamma markets w holdoucie: 10 654 single price, 2 016 TWAP30 i 15 550 TWAP60. Liczności te nie zostały ponownie wyliczone ani użyte do wyboru. Różnica czterech względem 28 224 slotów kalendarza nie wskazuje, które sloty nie mają potwierdzonego rynku; obecny pakiet tego nie mapuje. W przyszłej ocenie brakujące sloty otrzymają status unknown, nie zerowy PnL ani `no_trade`, i uniemożliwią pełny wynik.

Dotychczasowy T0009 replay i manifesty PMXT opisują development do `2026-07-01`, nie gotowy wejściowy zbiór E0041. Poprzedni raport reżimów wykazywał zero zweryfikowanych wierszy wykonania w okresie holdoutu. W tym zadaniu nie odczytano ani nie zwalidowano ksiąg, drabinek, fee, jakości quote’ów, outcome’ów ani gotowości T-60. Dokładne braki oraz pochodzenie liczb są w tabeli CSV.

E0039 F3 raw prefix jest plikiem mieszanym zarejestrowanym do `Opened=2026-06-30T23:59:00Z`. Runner przygotowania ma kopiować wyłącznie pełne wiersze `Opened < 2026-06-30T23:58:00Z`; wejście jest niebuforowane, a przy pierwszym wykluczonym wierszu odczytywany jest tylko leading `Opened` do separatora i reader zatrzymuje się przed OHLCV. W tym uruchomieniu gate zasobów ma poprzedzać jakikolwiek odczyt wejścia, więc jeszcze nie powstała nawet bezpieczna kopia developmentowa.

## Implementacja i testy

`run_btc_D_holdout_preparation_20261028.py` oraz `run_btc_D_holdout_evaluation_20261028.py` są osobnymi entrypointami. Przygotowanie sprawdza zapisane hashe kodu/protokołu, ogranicza raw reader, sprawdza zasoby i ma utrwalać selekcję, OOF Platt, finalny fit oraz parytet save/load wyłącznie na developmencie. Evaluator najpierw odczytuje protokół i odmawia, dopóki `evaluation_authorized` nie jest dokładnie `true`; przed tym guardem nie otwiera bundle’u, manifestu wejściowego ani danych. Źródło kwalifikuje dokładnie helper `utils.btc_t0009_source_audit.source_quarantine_reasons`; brakujące pola blokują źródło. Status source blocked, prediction missing, calibration missing, official outcome missing oraz znany `no_trade` są rozdzielone.

Weryfikacja przed fit: nowy zestaw testów syntetycznych 10/10; regresje E0040 księgi 10/10; chronologia E0039 20/20; pełne `python -m unittest discover -s tests` 759/759 w 45.068 s; `py_compile` i `git diff --check` przeszły. Pełny przebieg zgłosił istniejące `Pandas4Warning` i testowy `ResourceWarning`, bez błędów. Testy obejmują ścięcie mieszanego CSV przed payloadem, ścisłe granice etykiet i niezmienność maski po dopisaniu przyszłych wierszy, guard przed odczytem, źródłową symetrię, siatkę dziewięciu członków oraz ciągłość kont, terminalne wypłaty i miesięczne uzgodnienie przepływów na syntetycznych transakcjach. Nie jest to rzeczywisty fit E0041 ani walidacja treści holdoutu.

## Wynik bramki zasobów

Runner został zamrożony i wypchnięty jako commit `cb05ee640d80e9471c97eafdbe4f39164c77aeca` na `origin/audit/btc-dynamic-history-20261020` przed próbą przygotowania. Uruchomienie z 2026-10-28 o `12:26:31.352022 UTC` zakończyło się statusem `blocked_by_resource_gate` przed odczytem danych. Wolne miejsce wynosiło `69.64 GiB`, przy niezmienionym wymaganiu `80.0 GiB`; zapisany powód: `E0039 fit blocked by low free disk: 69.64 GiB`.

Gate odnotował `47.73 GiB` dostępnej z `63.67 GiB` pamięci, GPU `NVIDIA GeForce RTX 4060 Laptop GPU` (8188 MiB łącznie, 7344 MiB raportowane jako wolne), zero aktywnych procesów Python i czas gate `0.684785 s`. `run_identity_sha256` jest null; `holdout_rows_read=0`, `holdout_predictions_generated=0`. Pełny wynik zapisano lokalnie w `data/analysis/polymarket/BTC/btc_D_holdout_preparation_20261028/pre_fit_gate.json`; plik jest ignorowany przez Git. Nie otwarto raw prefixu, nie zaczęto selekcji/fitu/OOF/kalibracji i nie utworzono bundle’u.

Raport, tabela metadanych i aktualizacje rejestrów po gate zostały wypchnięte w commicie `1e80fd6b0ab9bd13175420c8ca40e66e4f92c0e9`.

## Uruchomienie i autoryzacja

Przygotowanie albo wznowienie po spełnieniu gate:

```powershell
python run_btc_D_holdout_preparation_20261028.py
```

Runner zapisuje pre-fit gate w `data/analysis/polymarket/BTC/btc_D_holdout_preparation_20261028/pre_fit_gate.json`; właściwe checkpointy etapów i manifest bundle’u powstają dopiero po przejściu gate i ukończeniu danego etapu. Fit można wznowić tym samym poleceniem po naturalnym spełnieniu zamrożonego progu. Nie obniżaj gate i nie usuwaj ani nie przenoś plików, by wymusić fit.

Późniejsza ocena wymaga osobnej, jawnej autoryzacji, gotowego `D_BUNDLE_READY` bundle’u oraz kompletnego `evaluation_rows.parquet` z manifestem i slotową inwentaryzacją. Dopiero po takiej autoryzacji zmień pole `evaluation.evaluation_authorized` w wersjonowanym protokole z `false` na `true`, po czym uruchom:

```powershell
python run_btc_D_holdout_evaluation_20261028.py
```

Tego pola nie zmieniono w przygotowaniu. Żadna wcześniejsza zgoda na retrospektywne analizy nie jest nową zgodą dla E0041. E0041 nie daje prawa do live/shadow, zleceń ani aktywacji modelu. Nie zamyka T0009 ani nie zmienia historycznego E0037 `FAILED`.

## Rejestr wykonania

Pierwszy pre-fit gate zablokował uruchomienie przy 69,64 GiB wolnego miejsca. Po bezpiecznym zastąpieniu czterech dokładnych duplikatów developerskich zweryfikowanymi hardlinkami odzyskano 19 934 871 552 B (18,5657 GiB), zachowując stare ścieżki oraz niezmieniony próg. Nie usuwano częściowego checkpointu, nie dotykano holdoutu ani T0046. Raport sprzątania i kolejnego uruchomienia: [szczegóły odzyskania miejsca i wyniku przygotowania](btc_D_holdout_preparation_20261028_storage_recovery_20261028.md).

Zamrożone polecenie `python run_btc_D_holdout_preparation_20261028.py` przeszło gate 2026-10-28 o 13:19:51.810052 UTC: 88,1991 GiB wolnego miejsca, 48,91 GiB dostępnego RAM, RTX 4060 z 7 247 MiB wolnej pamięci i 20 logicznych CPU. Run identity: `149ca92fac1ae8c5de3f14b74c30c8a26d9d82c8c1427f3fce08f84b43f49877`. Bezpieczny prefiks developerski skopiował 3 186 145 wierszy do 2026-06-30 23:57 UTC; reader zatrzymał się przed payloadem pierwszego wyłączonego timestampu 23:58. Holdout rows read: 0.

Etap wskaźników z sześcioma konfiguracjami trwał 400,144 s. Zamrożony etap profilu reakcji zakończył się po 2 704,55 s przy limicie 2 700 s, z 46/96 prób (44 complete, 2 pruned). Budżet jest kumulacyjny i wyczerpany; wznowienie bez zmiany wersji nie uruchomi dalszych prób, a zgodność ścieżki samplera po wznowieniu nie jest gwarantowana. Nie resetowano budżetu ani nie modyfikowano protokołu. Nie ukończono selekcji, nie wykonano finalnego fitu, OOF ani kalibracji i nie utworzono bundle’u. `evaluation_authorized=false`; `EVALUATION_CODE_READY`; `D_BUNDLE_BLOCKED_TUNING_BUDGET`; `HOLDOUT_NOT_RUN`. Nie czytano holdoutu ani nie generowano jego predykcji lub ekonomiki.

Testy ukierunkowane po uruchomieniu: trzy moduły E0041, rekonstrukcji chronologii i wznowienia strojenia — 43 testy oraz 52 podtesty zaliczone. Pełny zestaw 759/759 z raportu powyżej pochodzi sprzed tej próby. Wolne miejsce podczas aktywnego runu spadło do około 83,974 GiB, nadal powyżej progu. T0012 pozostaje BLOCKED z powodu zamrożonego limitu budżetu; kontynuacja wymaga osobnej, wersjonowanej zmiany protokołu i decyzji.

## Kontynuacja v2 zatwierdzona przed wznowieniem

Decyzja D0022 i protokół [E0041 v2](../../configs/research/btc_D_holdout_preparation_20261028_v2.json) zezwalają wyłącznie na wersjonowaną korektę budżetów: całe przygotowanie 72 000 → 76 500 s, selection chain 47 700 → 52 200 s, RP 2 700 → 7 200 s. Pozostałe limity, VP 2 700 s, kryterium 96 terminalnych prób i minimum pięciu poprawnych COMPLETE pozostają bez zmian. Zachowano osobny historyczny protokół [v1](../../configs/research/btc_D_holdout_preparation_20261028.json), jego manifest, SQLite study, historię budżetu i status niepowodzenia.

Study v1 ma 46 kolejnych terminalnych numerów (44 COMPLETE, 2 PRUNED); kontynuacja wymaga 50 nowych propozycji. RP wykorzystał 2 704,5527 s z 7 200, pozostawiając 4 495,4473 s. W całym limicie uwzględnia się 3 518,9517 s poprzedniego wykonania, a w selection chain 3 111,0753 s. Kopia SQLite przez API backup obejmuje zatwierdzony widok WAL; ukończone wskaźniki i bezpieczny prefiks są ponownie używane po weryfikacji hashy. Zapis stanu RNG samplera nie jest dostępny: `sampler_path_guaranteed_identical_after_resume=false`; TPE kontynuuje z zachowaną historią. `evaluation_authorized=false`.

Audyt istniejących czasów 46 prób: minimum 13,639 s, mediana 19,815 s, p90 33,231 s, maksimum 524,619 s, średnia 58,646 s, suma 2 697,721 s; cztery próby przekroczyły 400 s. Etap RP trwał 2 704,553 s, wskaźniki 400,144 s, całe v1 3 518,952 s, a selection chain 3 111,075 s. Profil używał jednego workera, 20 wątków i LightGBM GPU. Przegląd kodu potwierdził jednorazowe przygotowanie wspólnych tablic i foldów przed optymalizacją; każda próba nadal liczy zależne od parametrów cechy i objective. Nie znaleziono ponawianego pełnego odczytu wspólnych danych. Nie uruchomiono dodatkowych prób podczas audytu.

Bieżący odczyt zasobów przed wznowieniem: 83,97 GiB wolnego dysku, 46,55 GiB dostępnego RAM, 7 291 MiB wolnej pamięci GPU i 20 logicznych CPU. Aktywny T0048, PID 49516, używa 2,76 GiB; niezmieniona bramka blokuje uruchomienie, dopóki ten proces trwa. T0046/T0048 pozostają nietknięte. Kontynuacja nie została jeszcze uruchomiona; po spełnieniu gate należy wykonać polecenie bez argumentów z sekcji wyżej. Holdout rows read i predictions pozostają 0; `HOLDOUT_NOT_RUN`. E0037 pozostaje FAILED, T0009 BLOCKED.

Walidacja przed fit: ukierunkowane testy E0041 20/20; pełny zestaw `python -m unittest discover -s tests` 769/769 w 122,977 s; `py_compile`, parsowanie JSON v2, zgodność zamrożonych hashy kodu i `git diff --check` przeszły. Są to testy implementacji, nie walidacja referencyjna ani rzeczywisty fit, bundle lub wynik holdoutu.

Wersję v2 zamrożono i wypchnięto commitem `a6956c5` przed wywołaniem dokładnego polecenia `python run_btc_D_holdout_preparation_20261028.py`. Bramkowy zapis lokalny zawierał hash protokołu `d1f658b7e07c2284f0612f0f2144fdbea7dbe96c97df681ce9abfd0eea3f8f87`. Czas tej zablokowanej kontroli zasobów zostanie doliczony jeden raz do kumulacyjnego limitu całego przygotowania przy następnym przejściu gate; czas bezczynnego oczekiwania nie jest liczony. Dopracowanie rejestru gate nie zmienia limitów ani eksperymentu i zostanie wypchnięte przed wznowieniem.

Poprawka rejestru zachowuje historię każdej próby pre-fit: zablokowane kontrole zasobów doliczają się dokładnie raz do limitu całego przygotowania, a przejście gate jest już mierzone w aktywnej próbie. Czas oczekiwania między sesjami nie jest doliczany.

## Drugi wynik bramki zasobów v2

Po wypchnięciu poprawki kumulacyjnego rejestru gate w commicie `49c8ee0`, dokładne polecenie `python run_btc_D_holdout_preparation_20261028.py` ponownie zakończyło się statusem `blocked_by_resource_gate` o `2026-10-28T15:51:49.704048+00:00`. Kontrola trwała 1,1024812 s. Hash zamrożonego protokołu wynosi `1b289c7a24456171bf234dfa578ecc608bb9e68529b446972b3f9eb8b7414a1d`. Zasoby: dysk 83,97 GiB wolne przy wymaganych 80 GiB, RAM 46,27 GiB dostępne, GPU 7 271 MiB wolne z 8 188 MiB, 20 logicznych CPU. Aktywny T0048 PID 49516 używał 2,39 GiB; niezmieniona bramka procesu Python zatrzymała przygotowanie. T0046/T0048 pozostały nietknięte.

Pre-fit zapis zachował obie próby (1,1655582 s oraz 1,1024812 s; łącznie 2,2680394 s), każdą jeszcze niepoliczoną. Po przejściu bramki zostaną doliczone dokładnie raz do kumulacyjnego budżetu całego przygotowania; czas bezczynności pozostaje wyłączony. `run_identity_sha256=null`, liczba odczytanych wierszy holdoutu wynosi 0, liczba predykcji holdoutu wynosi 0. Nie rozpoczęto v2, fitu, selekcji ani odczytu nowego wejścia; nie powstał manifest v2 ani bundle. `evaluation_authorized=false`, `HOLDOUT_NOT_RUN`; E0037 pozostaje FAILED, a T0009 BLOCKED.

## Wznowienie v2 po przejsciu gate i blokada ponownego uruchomienia

Dokladne polecenie `python run_btc_D_holdout_preparation_20261028.py` przeszlo gate o `2026-10-28T15:57:45.037881+00:00` i wznowilo ten sam run identity `189049366841f6d025e3d9b080bb69dd0e441bb5638e451708f27c0635754c44`. Selekcyjny lancuch ukonczyl sie w 9 981.3369 s. RP dodal 50 prob do zachowanego prefiksu 46: wszystkie 96 sa terminalne (83 COMPLETE, 13 PRUNED), a minimum pieciu poprawnych COMPLETE jest spelnione. Nowy RP czas to 932.5 s; laczne uzycie v1+v2 wynosi 3 637.0527/7 200 s. VP zakonczyl sie w 1 357.4/2 700 s. Finalny zbior cech trwal 205.6 s, przyczynowy join 111.8 s, target weights 3 051.2/10 800 s, feature selection 2 400.6 s, a model tuning 1 877.8/7 500 s. Feature selection wskazal `recommended_k=256`, `best_k=320`; najlepszy model tuning to trial 41 z wartoscia 0.6922568800280187.

Po ukonczeniu etapow selekcyjnych runner pominol kalibracje ogolna i predykcje zewnetrzne zgodnie z trybem preparation-only, a nastepnie przerwal sie na `KeyError: 'minimum_successful_trials'` w `run_btc_D_holdout_preparation_20261028.py:1150`. Zapisany plik `v1_artifact_migration.json` nie przeniosl pola z zamrozonego protokolu. Uzupelniono wylacznie to pole wartoscia 5 z `budget_migration.minimum_successful_trials`; plik `v1_artifact_migration.repair_log.json` zachowuje hashe przed/po naprawie oraz hash protokolu. Kod, protokol i metoda eksperymentu nie zostaly zmienione.

Ponowne dokladne polecenie o `2026-10-28T18:46:54.898240+00:00` zakonczylo sie `blocked_by_resource_gate` po 0.6277065 s. Po wypchnieciu commitu dokumentacyjnego `6de64d0`, dokladne polecenie bez argumentow uruchomiono ponownie o `2026-10-28T18:51:03.242122+00:00`; gate ponownie zablokowal po 0.6476062 s. W obu przypadkach dysk mial 73.84 GiB wolnego przy wymaganych 80 GiB; w najnowszej kontroli RAM wynosil 51.26 GiB, GPU raportowalo 6 677 MiB wolnych. `run_identity_sha256=null`, odczytane wiersze holdoutu 0, predykcje holdoutu 0. Obie blokowane proby pre-fit (lacznie 1.2753127 s) zostana doliczone dokladnie raz do budzetu calego przygotowania po nastepnym przejsciu gate; bezczynnosc jest wylaczona. Manifest nadal wykazuje 13 511.5405/76 500 s uzyte i 62 988.4595 s pozostale przed doliczeniem tych prob.

Nie rozpoczeto finalnego dziewiecioczlonowego fitu D, koncowych OOF, kalibratora ani bundla. `evaluation_authorized=false`, `HOLDOUT_NOT_RUN`; holdout pozostaje nieodczytany, bez predykcji, scoringu i ekonomiki. E0037 pozostaje FAILED, T0009 BLOCKED. Wznowienie wymaga co najmniej 6.16 GiB dodatkowego wolnego miejsca na D:, bez obnizania progu ani przenoszenia/usuwania danych lub checkpointow.

## Kontynuacja po selekcji: checkpoint, miejsce i gate

Run zachowuje tozsamosc 189049366841f6d025e3d9b080bb69dd0e441bb5638e451708f27c0635754c44. Nie uzyto null identity z pozniejszych gate'ow jako dowodu braku runu. Protokol v2 ma SHA-256 1b289c7a24456171bf234dfa578ecc608bb9e68529b446972b3f9eb8b7414a1d; runner a89d67d2b802572003e28167ebbb6cc2b95793498d1dac593b7bcdb1d4a93181; naprawiony v1_artifact_migration.json d26082cacb88a006366470eed0a77b51a455fe55c3219b8df5d83620617f0139. Repair log potwierdza minimum_successful_trials=5 z zamrozonego protokolu oraz hashe przed/po.

Kontrola checkpointu przeszla. Manifest ma 96 terminalnych RP (83 COMPLETE, 13 PRUNED), RP, VP, target weights, feature selection i model tuning sa zakonczone. selection_result.json ma status fold_local_selection_completed i wiaze chronology_selection_result.json hashem b4f76f92923a5947faca2b333c8c583b522dc666ae9188579504bdfa58b7701c. W checkpointcie jest dokladnie 64 uporzadkowanych i unikalnych selected_base_features. model_feature_order to te 64 cechy, a nastepnie taker_Z_1m, taker_V_1m, taker_Z_5m, taker_V_5m, taker_Z_15m, taker_V_15m: lacznie 70. Ogolny selector pokazal recommended_k=256 i best_k=320, lecz tryb chronologiczny wymaga required_selected_feature_count=64. K=256/320 nie jest rozmiarem koncowego D; search pozostaje niezmieniony.

Zamrozony kod dokleja do BASE70 memory PCA8 i ESN PCA8 dla kazdego reservoir seed, czyli 86 kolumn na czlonka D. Kod sprawdza 8 nazwanych cech ESN dla kazdego z trzech seedow i kontrole liczby cech modelu po zapisie. Koncowy schemat nie zostal jeszcze zmaterializowany, poniewaz aktualny gate zatrzymal runner przed wznowieniem. Poprzedni log selection-only potwierdza pominiecie ogolnej kalibracji oraz zewnetrznych predykcji. Ostatnia proba gate nie doszla do sciezki resume; po jej przejsciu runner ma wczytac selection_result oraz pominac ukonczone search.

### Odzyskanie miejsca

Bezposrednio przed operacja D: mialo 79 181 156 352 B wolnego miejsca (73,743 GiB). Dwa historyczne, zakonczone artefakty developerskie zastapiono hardlinkami do sprawdzonych kopii kanonicznych. Obie sciezki nadal istnieja, a pelne SHA-256, rozmiar i os.samefile byly zgodne przed i po atomowej podmianie.

| Sciezka duplikatu z zachowana nazwa | Zachowana kopia kanoniczna | Bajty | SHA-256 |
|---|---|---:|---|
| E0039 v6: runs/eb07a734607052ac/F1_20260415_20260501/6cffa9ad0ec7a342/selection_pipeline/441e9566cadfbf70/stages/final_feature_dataset_4106633487b5/dataset/fold_20260415_20260501_preopen_v1.parquet | E0039 v7: runs/508858c48b11b925/F1_20260415_20260501/20914cfbc01dd0b3/selection_pipeline/441e9566cadfbf70/stages/final_feature_dataset_ddc936f8a38a/dataset/fold_20260415_20260501_preopen_v1.parquet | 4 438 416 358 | 3571d7678cb62bc29ca889a80d6b8a7b04a81eea4165ee3dcf56e2f3792df7b7 |
| calibration_seed: stages/final_feature_dataset_4527ce62c44f/dataset/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.parquet | calibration_seed: stages/final_feature_dataset_ecf685a9f3b0/dataset/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.parquet | 4 757 016 837 | dcefe1704eaf014ddf6f0616a24023b835ebd58f754476b4c101cd3cd29fa6fb |

Wolne miejsce wzroslo faktycznie o 9 191 477 248 B (8,5602 GiB), do 88 372 633 600 B (82,303 GiB). Nie usuwano danych. Nie ruszano recovery_archive, wejsc ani checkpointow E0041 ani znanego pliku tymczasowego 2 329 720 393 B z mozliwym checkpointem. Mtime nowego hardlinku odpowiada kopii kanonicznej; tozsamosc cache jest oparta o hash. Nie zatrzymywano procesow T0046/T0048 ani innych zadan.

Szacunek brakujacych wyjsc oparto o E0039 F3: 3 186 147 wierszy, dynamic_features.parquet 443 011 633 B, OOF predictions 5 660 209 B, 18 plikow modeli D 8 144 874 B. E0041 ma zblizona skale, wiec oczekiwane zapisane wyjscia sa ponizej 0,5 GiB; po ich zapisie powinno pozostac ok. 1,9 GiB ponad prog gate.

### Wynik gate po odzyskaniu miejsca

Dokladne polecenie python run_btc_D_holdout_preparation_20261028.py uruchomiono 2026-10-28T20:56:50.771682Z. Dysk spelnial prog 80 GiB: 82,30 GiB wolnego; RAM 46,08 GiB; GPU 6 731 MiB wolnej pamieci; 20 logicznych CPU. Niezmieniony gate zatrzymal wykonanie wylacznie z powodu PID 30900: python -X utf8 -m arbitron collect-betclic --concurrency 14, working set 2,64 GiB. Procesu nie zatrzymywano. Gate trwal 0,8948915 s, status blocked_by_resource_gate; proba jest zapisana jako jeszcze niepoliczona i wejdzie jeden raz do budzetu whole-preparation po pozniejszym przejsciu gate. Pozostale PID 44316 i 45856 rowniez pozostawiono bez zmian.

Proba zatrzymala sie przed odczytem wejsc: run_identity_sha256=null dla tego gate'u; holdout_rows_read=0 i holdout_predictions_generated=0. Wlasciwy preparation_manifest nadal identyfikuje poprzedni run powyzszym SHA. evaluation_authorized=false; nowe dynamic features, OOF, Platt i bundle nie powstaly. Progu nie obnizono. Wznowienie czeka na naturalne zejscie aktywnego procesu ponizej limitu; bezczynne oczekiwanie nie jest doliczane.

Kontrola zasobow o 2026-10-28T21:02:38Z wykazala 82,265 GiB wolnego na D: oraz 14,208 GiB working set PID 30900. Proces nadal dziala i pozostaje powyzej niezmienionego limitu; nie uruchomiono kolejnej proby runnera.

Odczyt o 2026-10-28T21:05:17Z wykazal spadek do 76,442 GiB wolnego na D: oraz aktywny PID 55988 z working set 9,761 GiB. Oba limity gate (80 GiB dysku i 1 GiB na Python workload) pozostaja niespelnione; runnera nie uruchomiono.

Odczyt o 2026-10-28T21:06:47Z nadal wykazywal 76,442 GiB wolnego, a PID 27580 mial 24,435 GiB working set. Nie uruchomiono runnera, bo oba progi resource gate pozostawaly niespelnione.
