# Przygotowanie przyszłej oceny holdoutu D

Data: 2026-10-28. Eksperyment E0041, zadanie T0012, decyzja D0021. Raport opisuje przygotowanie do jednego, z góry ustalonego przebiegu; nie jest raportem z wyniku holdoutu.

## Status

- `EVALUATION_CODE_READY`: runner oceny i testy mechaniki są gotowe; `evaluation_authorized` pozostaje `false`.
- `D_BUNDLE_READY`: **nie**. Fit D i Platt są jeszcze przed wykonaniem bramki zasobów.
- Dane: oficjalne Gamma i kalendarz rozstrzygnięć są `DATA_PRESENT_BY_METADATA`; T-60/T-59 w aktualnym pakiecie są `DATA_BLOCKED`; pokrycie OHLCV/przepływów, jakość treści, fee i mapowanie do wszystkich slotów są `DATA_NOT_VERIFIED`.
- `HOLDOUT_NOT_RUN`: prawda. Nie odczytano holdout payloadu, nie wygenerowano predykcji, nie uruchomiono scorera ani ekonomiki.

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

## Uruchomienie i autoryzacja

Przygotowanie albo wznowienie po spełnieniu gate:

```powershell
python run_btc_D_holdout_preparation_20261028.py
```

Runner zapisuje pre-fit gate w `data/analysis/polymarket/BTC/btc_D_holdout_preparation_20261028/pre_fit_gate.json`; właściwe checkpointy etapów i manifest bundle’u powstają dopiero po przejściu gate i ukończeniu danego etapu. Nie obniżaj gate i nie usuwaj ani nie przenoś plików, by wymusić fit.

Późniejsza ocena wymaga osobnej, jawnej autoryzacji, gotowego `D_BUNDLE_READY` bundle’u oraz kompletnego `evaluation_rows.parquet` z manifestem i slotową inwentaryzacją. Dopiero po takiej autoryzacji zmień pole `evaluation.evaluation_authorized` w wersjonowanym protokole z `false` na `true`, po czym uruchom:

```powershell
python run_btc_D_holdout_evaluation_20261028.py
```

Tego pola nie zmieniono w przygotowaniu. Żadna wcześniejsza zgoda na retrospektywne analizy nie jest nową zgodą dla E0041. E0041 nie daje prawa do live/shadow, zleceń ani aktywacji modelu. Nie zamyka T0009 ani nie zmienia historycznego E0037 `FAILED`.

## Rejestr wykonania

Status bramki zasobów, faktyczny czas, wynik push, pełny zestaw testów i końcowe SHA commitów zostaną dopisane po uruchomieniu pre-fit gate. Do tego momentu nie deklaruje się gotowości bundle’u.
