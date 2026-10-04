# BTC pre-open v1: trening nocny

Pipeline rozwija istniejący orchestrator `run_btc_preopen_experiment.py`. Korzysta z obecnych builderów cech, fitterów indicators/RP/VP, optymalizatora wag, selektora oraz strojenia LightGBM. Parametry przestrzeni wyszukiwania pozostają w dotychczasowych modułach. Aktywny profil to `nightly_v1` w [btc_preopen_training.json](../configs/btc_preopen_training.json), a kontrakt i zamrożony podział czasu są w [btc_preopen_v1.json](../configs/btc_preopen_v1.json).

## Uruchomienie i wznowienie

Z katalogu głównego repozytorium uruchom:

```powershell
python run_btc_preopen_experiment.py
```

To samo polecenie wznawia przerwane uruchomienie. Zgodne, ukończone etapy są pomijane po weryfikacji podpisów i sum kontrolnych artefaktów. Studia Optuny są przechowywane w SQLite i dopisują brakujące próby. Nieukończony fit indicators jest ponawiany, a kompletne fit-y tej rodziny pozostają do ponownego użycia. Dwa procesy nie mogą jednocześnie zapisywać tego samego uruchomienia.

Wyniki trafiają do `data/analysis/polymarket/BTC/preopen_v1/nightly_v1/<run-key>/`:

- `run.log` — główny log etapów, czasu, prób i błędów;
- `run_manifest.json` — postęp, podpisy wejść i zweryfikowane artefakty;
- `effective_config.json` — jedna konfiguracja użyta w całym uruchomieniu;
- `final_report.md` oraz `stages/external_evaluation_*/evaluation.json` — raport i metryki;
- `stages/external_evaluation_*/external_test_predictions.parquet` — predykcje;
- `stages/final_model_*/model_bundle.json` — manifest zestawu predykcyjnego.

Ścieżka bundle jest również zapisana jako `bundle_path` w `run_manifest.json`. Bundle wskazuje kontrakt, model, kalibrator, kolejność cech oraz konfiguracje i źródła generatorów. `load_prediction_bundle()` sprawdza kolejność cech i wczytuje model z kalibratorem. Bundle nie jest automatycznie aktywowany.

## Budżety i obliczenia

Profil ogranicza łączne uruchomienie do 12 godzin. Limity etapów wynoszą: indicators 6 h, RP 45 min, VP 45 min, wagi 2 h, selekcja 30 min, strojenie i końcowy model 45 min, kalibracja/ocena/bundle 15 min. Ich suma to 11 godzin; pozostała godzina jest marginesem całego uruchomienia. Limit sprawdzany jest między generacjami, próbami lub kandydatami. Termin etapu sprawdzany jest też po nieprzerywalnych fitach i weryfikacji artefaktów. Jeśli nie powstanie wymagany poprawny wynik, etap kończy się błędem i można wznowić go tym samym poleceniem.

Profil używa czterech workerów dla dopasowania indicators, czterech wątków CPU i GPU LightGBM. Wyszukiwania LightGBM działają sekwencyjnie, bez mnożenia workerów Optuny. Budżety wyszukiwania to: 6 rodzin indicators, populacja 64 i 8 generacji; po 24 próby RP i VP; 16 kandydatów wag; do 60 ocen selekcji cech; 40 prób modelu, do 500 estimatorów i early stopping po 50 rundach. Wewnętrzne podziały OOF są chronologiczne, a granice oceny pochodzą z kontraktu.

Lokalny pomiar: i7-13650HX (14 rdzeni/20 wątków), RTX 4060 Laptop (8 GB VRAM), Python 3.14.4 i LightGBM 4.6.0. Na 300 tys. wierszy i 96 cechach 100 rund trwało 1,20 s na GPU i 3,23 s na CPU (4 wątki). Integracyjny przebieg 8 tys. świec przeszedł całość w 21,5 s. Pełne wejście ma 3 321 148 wierszy; na tej podstawie oraz pomiarów krótkich operacji orientacyjny czas wynosi około 6–12 godzin. To szacunek, nie gwarancja; najwięcej czasu może zająć fit indicators na pełnej historii. Pełnego treningu nie uruchomiono.

## Kontrakt i ograniczenia oceny

Decyzja nominalna jest minutę przed startem rynku. Wiersz świecy `16:43` staje się dostępny o `16:44`; świece rozpoczynające się o `16:44` nie są używane do tej decyzji. Kontrakt rezerwuje do 45 sekund na obliczenie predykcji przed startem o `16:45`. Target porównuje `Open` z `16:45` z `Close` z `16:49`, a etykieta jest dostępna o `16:50`. Minuty pomocnicze zachowują własne przyszłe okna. Etykiety niedostępne na granicy fitowania i walidacji są odcinane; wcześniejsze świece pozostają do rozgrzewki cech.

Trening i kalibrator używają cenowego proxy Binance COIN-M BTCUSD Index. Ocena proxy działa bez danych wyników Polymarket. Metryki oficjalne są liczone tylko dla faktycznie dopasowanych etykiet rynku; replay ekonomiczny wymaga kwotowań sprzed startu. Okres testowy jest historycznie eksponowany w wcześniejszych analizach projektu i nie stanowi niezależnego, dziewiczego holdoutu.

Celowany smoke test uruchamia każdą z sześciu rodzin indicators, RP, VP, naliczenie cech, wagi, selekcję, tuning, końcowy fit, kalibrację, ocenę i próbę predykcji z zapisanego bundle. Jest to test integracji na syntetycznie datowanym fragmencie danych, nie wynik strategii. Uruchomienie smoke testu deweloperskiego: `python -c "import run_btc_preopen_experiment as r; r.run_integration_smoke()"`.
