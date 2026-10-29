# E0047 v4 - zapis nieudanej proby rewalidacji

Data: 2026-10-29. Protokol v4 SHA-256: `70c2e484bc3ab3f21323db595a5b622ed4456e2b235fd1c264dac0bd376665bf`. Commit prerejestracji: `7e8218531483238287562edac2eb079792bb2898`.

Uruchomienie bez argumentow CLI doszlo do zapisu overlap-tail po porownaniach E0046/E0047, po czym zakonczylo sie `OSError`: nie istnial katalog runtime `data/analysis/polymarket/BTC/btc_D_continuation_parity_closure_20261029_v4/suffixes`. Runner utworzyl katalog glowny runtime, ale nie jego katalog `suffixes` przed zapisem trzech plikow Parquet.

W runtime v4 pozostalo zero plikow; manifest, raport i parity CSV v4 nie powstaly. Ta proba nie ma przypisanych statusow komponentow i nie jest wynikiem audytu. Wszystkie hash-pinned pliki runtime/raportow v3 pozostaly bez zmian. Poprawka v5 tworzy katalog `suffixes` jawnie przed zapisem i zapisuje do nowego runtime.

Nie odczytano wartosci holdoutu, nie wygenerowano predykcji holdoutu i nie uruchomiono evaluatora, treningu, fitow, selekcji, live/shadow ani zlecen. `evaluation_authorized=false`, `activated=false`, `HOLDOUT_NOT_RUN` pozostaja bez zmian.
