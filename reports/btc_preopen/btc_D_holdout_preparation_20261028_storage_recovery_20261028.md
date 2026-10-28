# Odzyskanie miejsca i wynik przygotowania E0041

Data: 2026-10-28. Eksperyment E0041, zadanie T0012, decyzja D0021. Zakres obejmuje lokalne deduplikowanie dokładnych kopii artefaktów developerskich i uruchomienie zamrożonego przygotowania D. Nie obejmuje oceny holdoutu.

## Plan przed zmianą danych

Przed sprzątaniem wolumin D: miał 74 768 236 544 bajty wolnego miejsca (69,6333 GiB), czyli o 11 131 109 376 bajtów (10,3667 GiB) mniej niż wymagane przez zamrożony gate 80,0 GiB. Working tree był czysty na branchu `audit/btc-dynamic-history-20261020`, HEAD `b76c1b615b356ca407ae51ae699f3a8b5b658f5e`, upstream `origin/audit/btc-dynamic-history-20261020`.

Przed zmianą porównano metadane, manifesty i konfiguracje oraz strumieniowo sprawdzono SHA-256 plików z ukończonych runów. Cztery pary były bajtowo identyczne. W każdej parze stara ścieżka retry pozostała dostępna jako hardlink do pliku kanonicznego; manifesty i konfiguracje zachowały te same ścieżki.

| Run | Rozmiar duplikatu | SHA-256 | Plik kanoniczny pozostawiony | Ścieżka duplikatu zastąpiona hardlinkiem |
|---|---:|---|---|---|
| `dev_2026q2_a` | 4 571 824 016 B | `7532e24c7c28ff273d61b7139f55583179089b47c70c38a5756e1f20b16a2132` | `data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/shared_runs/dev_2026q2_a/224e784ef440f83b/stages/final_feature_dataset_6e0bb3e6567f/dataset/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.parquet` | `data/analysis/polymarket/BTC/preopen_research_20261011_full_compute20_retry/runs/dev_2026q2_a/224e784ef440f83b/stages/final_feature_dataset_55ef97f4f561/dataset/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.parquet` |
| `dev_2026q2_b` | 5 428 763 416 B | `162387794e0b612382177003585c34c140aa0b85aece44a1ee809f7cee9b973f` | `data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/shared_runs/dev_2026q2_b/cdf179bdec805886/stages/final_feature_dataset_25d78bd60067/dataset/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.parquet` | `data/analysis/polymarket/BTC/preopen_research_20261011_full_compute20_retry/runs/dev_2026q2_b/cdf179bdec805886/stages/final_feature_dataset_8e08241762b2/dataset/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.parquet` |
| `dev_2026q2_c` | 5 177 260 420 B | `e211d94ccc47227be2504a1c08d4019cb844d60ac6592c5fa6ec14aff512321c` | `data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/shared_runs/dev_2026q2_c/ade47faec2dc0b69/stages/final_feature_dataset_82d2eb6e4cae/dataset/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.parquet` | `data/analysis/polymarket/BTC/preopen_research_20261011_full_compute20_retry/runs/dev_2026q2_c/ade47faec2dc0b69/stages/final_feature_dataset_ec3969f605d2/dataset/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.parquet` |
| `calibration_seed` | 4 757 016 837 B | `dcefe1704eaf014ddf6f0616a24023b835ebd58f754476b4c101cd3cd29fa6fb` | `data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/shared_runs/calibration_seed/9ca4755fd7b45c9f/stages/final_feature_dataset_ecf685a9f3b0/dataset/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.parquet` | `data/analysis/polymarket/BTC/preopen_research_20261011_full_compute20_retry/runs/calibration_seed/9ca4755fd7b45c9f/stages/final_feature_dataset_77d629fa87a6/dataset/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.parquet` |

Nie użyto archiwizacji ani dodatkowej kompresji: pliki Parquet są już skompresowane. Para w `shared_runs/evaluated_holdout` i `runs/evaluated_holdout` została wyłączona; jej zawartości nie odczytano ani nie haszowano. Nie usunięto częściowego pliku `.BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.28244.tmp.parquet` (2 329 720 393 B), bo dokumentacja opisuje go jako checkpoint możliwy do wznowienia.

## Wynik sprzątania

Każdy hardlink i końcowa ścieżka po atomowej podmianie zostały sprawdzone przez SHA-256. Zachowano odnotowane czasy modyfikacji oryginalnych duplikatów.

| Run | Czas modyfikacji poprzedniej kopii UTC | Faktycznie odzyskane bajty | GiB | Hash końcowy |
|---|---|---:|---:|---|
| `dev_2026q2_a` | `2026-10-11T23:18:09.6156078Z` | 4 571 824 128 | 4,2578 | zgodny |
| `dev_2026q2_b` | `2026-10-12T01:14:12.3815889Z` | 5 428 764 672 | 5,0559 | zgodny |
| `dev_2026q2_c` | `2026-10-12T03:02:24.8449671Z` | 5 177 262 080 | 4,8217 | zgodny |
| `calibration_seed` | `2026-10-11T21:13:37.1132529Z` | 4 757 020 672 | 4,4303 | zgodny |

Łącznie zwolniono 19 934 871 552 bajty (18,5657 GiB). Po sprzątaniu D: miało 94 703 099 904 bajty wolnego miejsca (88,1991 GiB), o 8,1991 GiB ponad gate. Względem pomiaru początkowego wolnego miejsca przybyło 19 934 863 360 bajtów; różnica 8 192 bajtów to mały raport planu zapisany na tym samym woluminie.

Nietknięty checkpoint tymczasowy ma SHA-256 `b869fb5bf606fa354aeba35a9901bf7703e73d3a6b51c80214809fd7919d8d8d`. Nie zmieniano E0039/E0040, danych ani procesów T0046, aktywnych wejść E0041, modeli, checkpointów ani baz study. Nie zatrzymywano widocznego procesu `python -`.

## Gate i uruchomienie

Uruchomiono dokładnie `python run_btc_D_holdout_preparation_20261028.py`. Gate przeszedł 2026-10-28T13:19:51.810052Z: 88,1991 GiB wolnego miejsca wobec progu 80,0 GiB, 48,91 GiB dostępnego RAM, 7 247 MiB wolnej pamięci RTX 4060 i 20 logicznych CPU; `gate_failures=[]`. Istniejący proces `python -` (PID 44316, około 0,02 GiB) pozostawiono bez zmian. Podczas runu wolne miejsce spadło do około 83,974 GiB, nadal powyżej progu.

Run identity: `149ca92fac1ae8c5de3f14b74c30c8a26d9d82c8c1427f3fce08f84b43f49877`; katalog runu kończy się na `runs/149ca92fac1ae8c5`. Bezpieczny prefiks developerski zawiera 3 186 145 wierszy, 310 671 871 B, SHA-256 `25481546299f72c159ffe0c0b4f7e6c2c94b914b52973f9a49a29eb13427829a`, do `2026-06-30T23:57:00+00:00`. Pierwszy wyłączony timestamp to `2026-06-30T23:58:00+00:00`; reader zatrzymał się przed payloadem OHLCV. Odczyt wierszy holdoutu: 0.

Etap wskaźników zakończył się po 400,144 s dla sześciu konfiguracji. Etap profilu reakcji używał zamrożonego protokołu v1 i zakończył się limitem czasu: 2 704,55 s zużyte wobec 2 700 s, 46 z 96 prób terminalnych (44 complete, 2 pruned, 0 errors, 0 pending). Target 96 prób nie został osiągnięty. Manifest przygotowania ma status `interrupted_or_failed_with_checkpoints`; manifest etapu ma status `failed`. Checkpoint study SQLite zachowano.

Budżet jest kumulacyjny. `tuning_budget_history.json` zapisuje `budget_used_before_seconds=0`, `stage_elapsed_seconds=2704.5527`, `stage_budget_seconds=2700`, `budget_exhausted=true` i `sampler_path_guaranteed_identical_after_resume=false`. Kolejne uruchomienie nie zaczęłoby prób przy niezmienionym protokole. Nie resetowano historii, nie zmieniano limitu ani tożsamości runu. Kontynuacja wymaga wersjonowanej zmiany protokołu i odpowiedniej decyzji; samo zwiększenie wolnego miejsca nie usuwa tej blokady.

Nie ukończono selekcji ani finalnego fitu. Nie ma OOF, kalibratora Platta ani bundle’u D. `evaluation_authorized=false`; `EVALUATION_CODE_READY`; `D_BUNDLE_BLOCKED_TUNING_BUDGET`; `HOLDOUT_NOT_RUN`. Nie odczytano holdoutu, nie wygenerowano jego predykcji ani ekonomiki. Stan E0037 `FAILED`, T0009 `BLOCKED` i pozostałe niepowiązane zadania nie zmieniły się.

Testy ukierunkowane po runie: `tests/test_btc_D_holdout_preparation_20261028.py`, `tests/test_btc_chronology_reconstruction_20261027.py` i `tests/test_btc_preopen_tuning_resume.py` — 43 testy oraz 52 podtesty zaliczone.

Raport i aktualizacje kanonicznych rejestrów są objęte wspólnym commitem dokumentacyjnym; SHA i wynik push podano w komunikacie końcowym.

## Dodatkowe odzyskanie miejsca po selekcji E0041 v2 - 2026-10-28

Przed operacja wolumin D: mial 79 181 156 352 B wolnego (73,743 GiB). Pelne SHA-256 sprawdzono po obu stronach kazdej pary. Sciezke docelowa zastapiono po weryfikacji tymczasowego hardlinku atomowym os.replace; po operacji potwierdzono ten sam plik, rozmiar i hash. Obie historyczne sciezki zachowano.

| Sciezka zlinkowana | Zachowana kopia kanoniczna | Rozmiar | SHA-256 |
|---|---|---:|---|
| data/analysis/polymarket/BTC/btc_chronology_reconstruction_20261027/runs/eb07a734607052ac/F1_20260415_20260501/6cffa9ad0ec7a342/selection_pipeline/441e9566cadfbf70/stages/final_feature_dataset_4106633487b5/dataset/fold_20260415_20260501_preopen_v1.parquet | data/analysis/polymarket/BTC/btc_chronology_reconstruction_20261027/runs/508858c48b11b925/F1_20260415_20260501/20914cfbc01dd0b3/selection_pipeline/441e9566cadfbf70/stages/final_feature_dataset_ddc936f8a38a/dataset/fold_20260415_20260501_preopen_v1.parquet | 4 438 416 358 B | 3571d7678cb62bc29ca889a80d6b8a7b04a81eea4165ee3dcf56e2f3792df7b7 |
| data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/shared_runs/calibration_seed/9ca4755fd7b45c9f/stages/final_feature_dataset_4527ce62c44f/dataset/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.parquet | data/analysis/polymarket/BTC/btc_preopen_model_update_20261012/shared_runs/calibration_seed/9ca4755fd7b45c9f/stages/final_feature_dataset_ecf685a9f3b0/dataset/BTCUSD_INDEXVOL_UM_BTCUSDT1m_research_cutoff_20261007_preopen_v1.parquet | 4 757 016 837 B | dcefe1704eaf014ddf6f0616a24023b835ebd58f754476b4c101cd3cd29fa6fb |

Rzeczywisty przyrost free space: 9 191 477 248 B (8,5602 GiB), z 79 181 156 352 B (73,743 GiB) do 88 372 633 600 B (82,303 GiB). Mtime sciezki po hardlinku odpowiada plikowi kanonicznemu: target F1 1793121638263857300 ns -> 1793125445990803500 ns; target calibration_seed 1791848926323342700 ns -> 1791855120065110600 ns. Hash jest podstawa tozsamosci danych/checkpointow. Nie usuwano plikow; nie ruszano recovery_archive, danych/checkpointow/studies E0041 ani pliku tymczasowego 2 329 720 393 B o znanym mozliwym wznowieniu. Procesow T0046/T0048 ani innych zadan nie zatrzymano.

Szacunek pozostalych wyjsc z ukończonego E0039 F3 o podobnej liczbie wierszy: dynamic feature parquet 443 011 633 B dla 3 186 147 rows, inner OOF predictions 5 660 209 B, 18 plikow modeli D 8 144 874 B. Lacznosc outputow E0041 szacowana ponizej 0,5 GiB, pozostawiajac ok. 1,8 GiB ponad gate.

Gate po odzyskaniu: exact no-argument command z 2026-10-28T20:56:50.771682Z, disk 82,30 GiB free, RAM 46,08 GiB, GPU 6 731 MiB free, 20 logical CPUs. Proces PID 30900, python -X utf8 -m arbitron collect-betclic --concurrency 14, uzywal 2,64 GiB i zablokowal unchanged workload threshold przed input access. Process pozostawiono w spokoju. Gate elapsed 0,8948915 s, blocked_by_resource_gate, run identity null, holdout rows/predictions zero. Ta proba jest jeszcze niepoliczona i zostanie dodana raz po przyszlym gate pass; idle time jest wykluczony.

Ostatnia kontrola 2026-10-28T21:02:38Z: D: mialo 82,265 GiB wolnego, a PID 30900 mial 14,208 GiB working set. Proces nadal dzialal; nie uruchomiono runnera ponownie, bo niezmieniony limit Python workload nie byl spelniony.

Pomiar 2026-10-28T21:05:17Z wykazal 76,442 GiB wolnego i aktywny PID 55988 z working set 9,761 GiB; oba warunki resource gate pozostaja niespelnione.

Odczyt 2026-10-28T21:06:47Z nadal wykazywal 76,442 GiB wolnego, a PID 27580 mial working set 24,435 GiB.

Ostatni pomiar 2026-10-28T21:11:54Z: 76,441 GiB wolnego; pozostale PID 44316 i 45856 mialy po 0,012 GiB working set. Brakowalo 3,559 GiB do progu dyskowego 80 GiB, wiec nie uruchomiono runnera.
