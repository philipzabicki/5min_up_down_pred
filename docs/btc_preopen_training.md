# BTC pre-open v1: trening nocny

Pipeline rozwija istniejący orchestrator `run_btc_preopen_experiment.py`. Korzysta z obecnego buildera danych, fitterów indicators/RP/VP, optymalizatora wag, selektora oraz strojenia LightGBM. Przestrzenie wyszukiwania pozostają w dotychczasowych modułach. Aktywny profil `nightly_v1` znajduje się w [btc_preopen_training.json](../configs/btc_preopen_training.json), a target i zamrożony podział czasu w [btc_preopen_v1.json](../configs/btc_preopen_v1.json).

## Uruchomienie i wznowienie

Z katalogu głównego repozytorium uruchom:

```powershell
python run_btc_preopen_experiment.py
```

To samo polecenie wznawia przerwane uruchomienie. Etapy są pomijane po sprawdzeniu sygnatur wejść i sum kontrolnych artefaktów. Studia Optuny są przechowywane w SQLite. Jeśli budżet zatrzyma wyszukiwanie po osiągnięciu minimum poprawnych wyników, najlepszy wynik zostaje zapisany, a niezmieniony resume pomija ukończony etap, nawet gdy nie osiągnięto celu prób. Zmiana konfiguracji wyszukiwania tworzy nową sygnaturę etapu. Niekompletny fit indicators jest wznawiany; zmiana implementacji wskaźnika unieważnia również wewnętrzny cache fittera. Tylko jeden proces może jednocześnie zapisywać do tego samego katalogu uruchomienia.

Wyniki trafiają do `data/analysis/polymarket/BTC/preopen_v1/nightly_v1/<run-key>/`:

- `run.log` — główny log etapów, czasu, prób i błędów;
- `run_manifest.json` — postęp, sygnatury i zweryfikowane artefakty;
- `effective_config.json` — efektywna konfiguracja uruchomienia;
- `final_report.md` oraz `stages/external_evaluation_*/evaluation.json` — raport i metryki;
- `stages/external_evaluation_*/external_test_predictions.parquet` — predykcje;
- `stages/final_model_*/model_bundle.json` — manifest zestawu predykcyjnego.

Ścieżka bundle jest zapisana jako `bundle_path` w `run_manifest.json`. Przy wznowieniu orchestrator odtwarza ścieżki datasetu, stanów RP/VP, konfiguracji generatorów, kolejności cech, modelu i kalibratora z wyników etapów. Brak wymaganej zależności zatrzymuje tworzenie bundle czytelnym błędem. Bundle nie jest automatycznie aktywowany.

Weryfikacja bundle działa w świeżym procesie. Odtwarza wybrane cechy istniejącą ścieżką buildera z chronologicznego prefiksu surowych świec i świeżego stanu generatorów, następnie porównuje predykcję modelu i kalibratora z odpowiadającą jej predykcją ewaluacyjną treningu. Jawna tolerancja bezwzględna wynosi `1e-6`; wynik trafia do `raw_replay_verification.json`. Resume pomija tę weryfikację, gdy jej artefakt i zależności kodowe nadal są zgodne.

## Budżety i obliczenia

Profil dopuszcza do 14 godzin. Limity etapów wynoszą: indicators 6 h, RP 45 min, VP 45 min, wagi 2 h, selekcja 30 min, strojenie i końcowy fit 45 min, kalibracja/ocena/weryfikacja bundle 2 h. Ich suma to 12 h 45 min, zatem pozostaje 1 h 15 min marginesu całościowego.

Limit wyszukiwania jest sprawdzany między generacjami, próbami, foldami lub kandydatami. Pojedynczy fit lub replay surowych danych może przekroczyć budżet, jeśli rozpoczął się przed terminem. `_run_stage` nie odrzuca poprawnego wyniku tylko dlatego, że termin upłynął podczas jednostki pracy lub zapisu artefaktów. Indicators wymagają skonfigurowanego minimum generacji (domyślnie jednej); RP/VP i tuning modelu — minimum poprawnych prób; strojenie wag — co najmniej trzech ocenionych wag albo całego mniejszego zbioru kandydatów; selekcja — minimalnej liczby foldów prescreeningu i trzech kompletnych ocen top-K. Nieosiągnięcie minimum zatrzymuje przebieg i zachowuje dostępne checkpointy. Dataset, model i bundle są ukończone dopiero po zapisaniu i sprawdzeniu wymaganych artefaktów.

Profil używa czterech workerów dla fittera indicators i czterech wątków LightGBM. Wyszukiwania LightGBM działają sekwencyjnie, z jednym workerem Optuny. Cele wyszukiwania to sześć rodzin indicators z populacją 64 i 8 generacjami; po 24 próby RP i VP; 16 kandydatów wag; do 60 ocen selekcji cech; 40 prób modelu, do 500 estimatorów i early stopping po 50 rundach. Są to limity pracy, a nie gwarancja czasu pełnego treningu. Pełnego przebiegu na całej historii 3 321 148 wierszy nie uruchomiono ani nie zmierzono w ramach tej zmiany.

Selekcja cech i tuning modelu trenują każdy chronologiczny fold na wszystkich kwalifikujących się minutach przed walidacją, po odcięciu etykiet niedostępnych przy pierwszej decyzji walidacyjnej. Wagi obserwacji to `decision_weight` dla minut decyzyjnych oraz `(1 - decision_weight) / 4` dla minut pomocniczych. Walidacja, early stopping, ranking, permutacja i ocena kandydatów używają wyłącznie rzeczywistych minut decyzyjnych, bez wag obserwacji. Istniejące wagi agregacji foldów pozostają odrębne.

## Kontrakt i ograniczenia oceny

Decyzja nominalna przypada minutę przed startem rynku. Świeca `16:43` staje się dostępna o `16:44`; świece otwierające się o `16:44` są wykluczone z tej decyzji. Kontrakt rezerwuje do 45 sekund na predykcję przed startem o `16:45`. Target porównuje `Open` z `16:45` z `Close` z `16:49`, a etykieta jest dostępna o `16:50`. Minuty pomocnicze zachowują własne przyszłe okna. Etykiety niedostępne na granicy treningu i walidacji są odcinane; wcześniejsze świece pozostają do rozgrzewki cech.

Trening i kalibrator używają cenowego proxy Binance COIN-M BTCUSD Index, a nie oficjalnego rozliczenia Polymarket. Oficjalne metryki wymagają dopasowanych etykiet rynku; replay ekonomiczny wymaga kwotowań zarejestrowanych przed startem. Okres testowy był używany we wcześniejszych analizach projektu i nie stanowi niezależnego, nietkniętego holdoutu.

Integracyjny smoke test używa syntetycznie datowanego fixture 8 000 świec, wykonuje pipeline i jego wznowienie, sprawdza pominięcie ukończonych etapów oraz odtwarza bundle z surowych świec. To kontrola techniczna, nie wynik strategii. Uruchomienie:

```powershell
python -c "import run_btc_preopen_experiment as r; r.run_integration_smoke()"
```
