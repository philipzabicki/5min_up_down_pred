# BTC pre-open: trzy metody polityki wejścia, sizingu i wyjścia

Data: 2026-10-07. Status: **specyfikacja badawcza; metody nie zostały wdrożone ani uruchomione przez dodanie tego dokumentu**.

## 1. Cel i wybór metod

Sprawdzić trzy konkretne alternatywy dla obecnej polityki:

1. **Distributionally robust Kelly (DRK)** — wybór strony, brak wejścia i wielkość zakładu z uwzględnieniem niepewności prawdopodobieństwa.
2. **Risk-constrained Kelly (RCK)** — wybór strony i stawki przy jawnym ograniczeniu ryzyka kapitału oraz otwartych pozycji.
3. **Optimal stopping (OS)** — decyzja, czy sprzedać posiadaną pozycję, czy kontynuować do kolejnego momentu lub rozliczenia.

DRK i RCK porównać najpierw oddzielnie. OS ocenić najpierw na zamrożonych wejściach i stawkach, następnie jako pełną politykę portfelową z uwalnianiem gotówki. Łączenie zwycięskich elementów jest osobnym eksperymentem, a nie czwartą rodziną przeszukiwaną od początku.

Są to uzasadnione literaturą metody do adaptacji. Nie twierdzimy, że którakolwiek jest potwierdzonym SOTA na pięciominutowym BTC/Polymarket. MPC pomijamy w pierwszej trójce: wymaga szerszego modelu dynamiki rynku i sekwencji transakcji. Deep RL również nie jest potrzebny do pierwszej implementacji.

## 2. Punkt wyjścia zweryfikowany w kodzie

Odniesienie: commit [68f69ad](https://github.com/philipzabicki/5min_up_down_pred/commit/68f69adb5f6fe2b93580f216c80e27170f359a76); pliki odczytano również z main przy przygotowaniu dokumentu:

- [run_btc_preopen_policy_optimization.py](../run_btc_preopen_policy_optimization.py)
- [konfiguracja dotychczasowego badania](../configs/research/btc_preopen_policy_search_20261007.json)
- [raport](../reports/btc_preopen/SUMMARY.md)

Istotne funkcje i ograniczenia:

| Element | Obecne zachowanie | Konsekwencja dla nowego badania |
| --- | --- | --- |
| _trial_grid | 58 konfiguracji: stała stawka, procent gotówki, fractional Kelly, brak transakcji | Wynik dotyczy tej przestrzeni, nie całej klasy polityk |
| _sized_request | Kelly z punktowego p_candidate_platt, limitów i cost-basis equity | Brak jawnego modelu niepewności p i wspólnego ryzyka wypłat |
| _side_candidate | Filtr zwrotu i dodatniego oczekiwanego log-growth dla konkretnej kwoty | Zmiana kwoty może zmieniać przyjęcie transakcji |
| _feasible_order | Gotówka, ilość na best ask, minima, zaokrąglenia i limit $20 | Zachować wspólną semantykę wykonania i wyraźne powody odrzucenia |
| Portfolio.equity | Wolna gotówka + koszt nierozliczonych zakupów | Nie jest to wartość likwidacyjna ani losowa końcowa wypłata portfela |
| matched_fixed5 w run | Kopiuje próg wybranej polityki, ponownie wyznacza transakcje | Nie izoluje samego sizingu na identycznych transakcjach |
| Artefakt decyzji | Jeden zakup T−59, hold_to_resolution=True | Wyjścia nie były optymalizowane |

Nie zmieniać historycznych wyników w miejscu. Poprzednie artefakty pozostają punktem odniesienia, nowe mają własną wersję, konfigurację i fingerprinty.

## 3. Wspólny kontrakt ekonomiczny i danych

### 3.1. Decyzja wejścia

- Jedyny moment pierwszego wejścia: **T−59 s**, gdzie T to początek docelowego rynku 5-minutowego.
- Nie optymalizować opóźnienia ani alternatywnych momentów pierwszego wejścia w tym badaniu.
- Akcje: brak zakupu, zakup UP lub zakup DOWN. Maksymalnie jedna strona na rynek.
- Predykcja bazowa: zamrożony candidate_platt. Nie trenować ponownie bazowego modelu BTC w ramach implementacji tych polityk.
- Informacje są dostępne według czasu odbioru, a etykiety według czasu ich faktycznej dostępności.
- ask age jest właściwością aktualizacji booka, nie opóźnieniem wejścia. Nie dodawać osobnego strojenia progów ask age.
- Dziedziczyć obecną kwalifikację booków, opłaty i założenia minimalnego zamówienia; oddzielić założenia badawcze od potwierdzonych reguł historycznych.

### 3.2. Wykonanie i kapitał

Wspólny początkowy kapitał $100, brak kredytu i istniejący techniczny limit $20 na zakup. Limit jest warunkiem porównania, nie twierdzeniem o optymalnej stawce.

W pierwszym porównaniu zakupy ogranicza zapisana ilość na najlepszym asku. Nie wymagać pełnej drabinki jako warunku rozpoczęcia DRK/RCK. Zachować dokładny historyczny replay $5 jako dodatkową kontrolę, ale porównywać nowe polityki z baseline'em również w tym samym modelu najlepszego poziomu.

Dla każdej wykonalnej kwoty obliczać osobno:

- s: zakup brutto w USD;
- D(s): pełny debet gotówki, w tym opłata gotówkowa;
- Q(s): liczba netto otrzymanych udziałów, po ewentualnej opłacie udziałowej;
- przy wygranej wypłata Q(s), przy przegranej 0.

Po zaokrągleniu kwoty ponownie obliczyć opłaty, minima, cel i ograniczenia. Nie stosować poprawnego wzoru ciągłego do niewykonalnej, zaokrąglonej transakcji.

Gotówkę z rozliczenia udostępniać według istniejącego kontraktu +60 s. Gotówkę ze sprzedaży udostępniać według jawnego modelu wykonania i rozliczenia sprzedaży; nie stosować automatycznie opóźnienia redemption. Założenie oznaczyć w wynikach.

Offline nie ustala prawdziwego fillu. Wymagamy spójnego modelu ceny, ilości, opłat i czasu dostępności, wspólnego dla porównywanych metod. Nie blokować badania małych stawek wymaganiem idealnego symulatora wykonania.

### 3.3. Trzy różne wartości portfela

Raportować oddzielnie:

1. **Gotówkę** — realne ograniczenie nowych zakupów.
2. **Cost-basis equity** — zgodność z poprzednimi raportami.
3. **Wartość likwidacyjną** po dostępnym bidzie, kosztach i ilości — gdy istnieją odpowiednie dane.

Brak bidu/ilości oznacza nieznaną wykonalną wartość, nie wycenę po asku lub automatycznie po koszcie. Ryzyko końcowej wypłaty modelować przez scenariusze wyniku, a nie przez koszt zakupu.

## 4. Metoda 1: distributionally robust Kelly

### 4.1. Hipoteza

Część dużych nominalnych przewag p względem ceny wynika z błędu estymacji lub kalibracji. Uwzględnienie tego błędu może poprawić sizing i odrzucanie transakcji względem stałego mnożnika Kelly'ego.

### 4.2. Definicja

Dla pojedynczego zakładu, bez innych losowych pozycji, z kapitałem W:

G(s,p) = p log((W − D(s) + Q(s))/W)
       + (1 − p) log((W − D(s))/W).

Wybieramy stronę i wykonalne s, maksymalizując:

max_s min_{p ∈ [p_low, p_high]} G(s,p).

Brak transakcji daje 0. Jeśli najlepszy wynik nie jest dodatni, nie kupujemy. Wymagamy dodatniego majątku we wszystkich dopuszczonych wynikach.

Dla zakupu UP najgorszy p to p_low. Dla DOWN przedział wynosi [1 − p_high, 1 − p_low]. Dla dodatniej wypłaty wzrost jest monotoniczny względem prawdopodobieństwa wygranej, więc nie potrzeba zagnieżdżonego optymalizatora po p.

Kontrola bez opłat, przy cenie udziału a i stawce jako odsetku kapitału:

f* = max(0, (p_low − a)/(1 − a)).

To test analityczny szczególnego przypadku, nie zamiennik pełnego rachunku D/Q.

Przy istniejących pozycjach stosować scenariusze wspólnych wypłat opisane w sekcji 5.3. Wówczas celem jest najgorsza oczekiwana logarytmiczna wartość końcowego majątku względem dopuszczonych wag scenariuszy. Nie podstawiać kosztu otwartych pozycji jako pewnej wypłaty. Wersję binarną traktować jako kontrolę izolowanego zakładu.

### 4.3. Skąd przedział niepewności

Pierwsza implementacja: prosty model kalibracyjny p, dopasowany wyłącznie na przeszłych, dostępnych etykietach; kandydaci wejścia to logit bazowego p oraz bieżąca cena rynku. Cena jest informacją wejściową, nie automatycznie prawdziwym prawdopodobieństwem.

- Użyć regularizowanej regresji logistycznej jako pierwszego modelu.
- Porównać wersję opartą tylko na p z wersją zawierającą również cenę; wybór wyłącznie w wewnętrznej walidacji.
- Estymować rozrzut przez bootstrap bloków czasu, np. 3-dniowych, z wcześniejszego zbioru; długość i liczbę replik zamrozić przed oceną.
- Przedział kwantyli predykcji bootstrapowych stanowi empiryczny zbiór niepewności modelu. Nie nazywać go gwarantowanym przedziałem dla prawdziwego warunkowego p.
- Przy zbyt małej historii użyć jawnie określonego globalnego wariantu kalibracji albo braku transakcji. Regułę i minimalną liczbę bloków ustalić przed wynikami.
- Strojenie obejmuje siłę regularizacji, zakres historii i szerokość zbioru niepewności. Nie wyznaczać ich na outer test.

Kontrole: bazowy p; ten sam nowy punktowy p bez niepewności; DRK na tym samym p. Dzięki temu poprawa kalibracji nie zostanie błędnie przypisana odporności Kelly'ego.

### 4.4. Integracja i logi

Rozszerzyć obliczanie kandydatów oraz sizing, zachowując wspólny rachunek wykonania. Logować p bazowe, p centralne, przedział UP, stronę, oczekiwany i najgorszy log-growth, stawkę przed/po ograniczeniach oraz przyczynę braku wejścia.

Kryterium badawcze: lepszy wzrost przy porównywalnym ryzyku lub mniejsze ryzyko przy porównywalnym wzroście, utrzymujące się w kolejnych blokach.

## 5. Metoda 2: risk-constrained Kelly

### 5.1. Hipoteza i różnica względem DRK

DRK chroni przed niepewnością oszacowania przewagi. RCK ogranicza ryzyko trajektorii majątku. W pierwszym porównaniu RCK używa tego samego punktowego p co odpowiednia kontrola Kelly'ego, aby nie mieszać obu mechanizmów.

### 5.2. Oryginalny problem i jego granice

Niech alpha ∈ (0,1) oznacza poziom majątku względem startu, a beta ∈ (0,1) dopuszczalne prawdopodobieństwo zejścia poniżej tego poziomu. W konstrukcji Busseti–Ryu–Boyd definiuje się:

lambda = log(beta) / log(alpha) > 0.

Maksymalizujemy E[log R] przy ograniczeniu E[R^(−lambda)] ≤ 1, gdzie R jest mnożnikiem majątku jednego kroku. Odpowiednie założenia procesu i warunkowych ograniczeń pozwalają ograniczyć ryzyko zejścia poniżej alpha.

Nie jest to gwarancja maksymalnego drawdownu od ruchomego szczytu. Nie przenosić automatycznie gwarancji na zależne, nakładające się zakłady, błędne p i przybliżony replay.

### 5.3. Implementacja dla portfela

Dla każdej akcji obliczyć majątek końcowy we wspólnym horyzoncie rozliczenia obecnych pozycji i nowego zakupu:

W_j(s) = C − D(s) + Σ_i q_i Y_ij + Q(s) Y_new,j,

gdzie C to aktualna gotówka, q_i to posiadane udziały netto, Y_ij ∈ {0,1}, a j oznacza scenariusz wspólnych wyników. Uwzględnić także już należne, lecz jeszcze zablokowane środki bez podwójnego liczenia.

- Prognozowane wagi scenariuszy pi_j wyznaczać wyłącznie z wcześniejszych danych.
- Nie zakładać niezależności sąsiednich rynków bez kontroli. Zastosować wspólne scenariusze z czasowego resamplingu zależności błędów oraz porównać wariant niezależny.
- Przy małej liczbie pozycji enumerować kombinacje wyników; przy większej użyć wspólnego, deterministycznie seedowanego banku scenariuszy.
- Opisać mapowanie błędów/prognoz na scenariusze i sprawdzić odtworzenie założonych prawdopodobieństw brzegowych.
- Brak transakcji pozostaje akcją dopuszczalną. Nie wymuszać kolejnego zakupu, aby naprawić ryzyko już otwartego portfela.

W pierwszej wersji z nakładającymi się pozycjami maksymalizować Σ_j pi_j log W_j(s), a ograniczenie potęgowe liczyć dla R_j = W_j(s)/W_ref. W_ref to bieżąca wartość likwidacyjna portfela, jeśli da się ją odtworzyć; w wariancie tylko z cache wejść użyć cost-basis equity i wyraźnie oznaczyć ten przybliżony wariant. W_ref ustalić przed oceną akcji i nie zmieniać między kandydatami. Nie używać oczekiwanej końcowej wypłaty jako W_ref: przez wypukłość potęgi ujemnej mogłoby to odrzucać nawet brak transakcji dla każdej niezerowej niepewności.

**To portfelowa adaptacja/ograniczenie lokalne, nie udowodniona gwarancja wielookresowa z publikacji.** Dokładną wersję teoretyczną weryfikować oddzielnie na niepokrywających się zakładach. Jeżeli nawet brak transakcji nie spełnia lokalnego ograniczenia, nie dodawać ekspozycji; zapisać naruszenie, nie deklarować gwarancji.

Warunki wykonalności nadal obejmują gotówkę teraz, ilość, opłaty, minima i limit kwoty. Dodatnie oczekiwane przyszłe wypłaty nie finansują bieżącego zakupu.

### 5.4. Parametry i ocena

Stroić intensywność ograniczenia ryzyka, nie równocześnie dowolne mnożniki Kelly'ego. Dla przykładowych interpretowalnych profili można przyjąć alpha = 0.5/0.7/0.8 i beta = 0.05/0.10/0.20; wartości są punktami badawczymi, nie zaakceptowanym budżetem ryzyka użytkownika. Usunąć duplikaty lambda.

Raportować empirycznie: wzrost, najgorszy wynik blokowy, drawdown kosztowy i dostępny likwidacyjny, ekspozycję, czas pod wodą, częstość naruszeń i liczbę odrzuceń przez ograniczenie ryzyka. Nie przeliczać empirycznej częstości na gwarancję likwidacji.

Kontrole: ta sama predykcja i scenariusze bez ograniczenia; fractional Kelly; DRK oddzielnie. Zestawić krzywe wzrost–ryzyko, a nie wyłącznie jednego zwycięzcę PnL.

## 6. Metoda 3: optimal stopping dla wyjść

### 6.1. Zakres

Pierwsze wejście nadal następuje T−59. Pierwsza wersja wyjścia ma dwie akcje: sprzedaż całej posiadanej pozycji lub dalsze trzymanie. Nie dodawać ponownych wejść, uśredniania, zmiany strony i częściowej sprzedaży w tej wersji.

Pozwala to zbadać wartość wyjść bez równoczesnego rozszerzania przestrzeni wszystkich możliwych transakcji.

### 6.2. Wymagane trajektorie

Dla każdego rynku od wejścia do końca notowań/rozliczenia potrzeba:

- przyczynowo odtworzonych bidów, asków, ilości i czasów odbioru;
- stanu pozycji, gotówki, opłat oraz czasu do końca;
- dostępnych wtedy informacji o BTC i ewentualnej aktualnej prognozy;
- zdarzenia końcowego wyniku i jego dostępności.

Sam cache wejścia i końcowa etykieta nie wystarczają. Najpierw sprawdzić zapisane lokalne dane i manifesty. Jeśli trzeba odtworzyć trajektorie, przygotować jeden współdzielony cache, z zakresem, szacowanym czasem i postępem; nie skanować archiwum osobno dla każdej polityki.

Istniejącego modelu T−59 nie wywoływać na późniejszych świecach jako rzekomo tego samego zadania. Pierwsza polityka może używać zamrożonego p z wejścia i aktualnego stanu rynku. Opcjonalny model aktualnego prawdopodobieństwa musi mieć poprawny target i czas pozostały do rozliczenia, oraz osobną walidację.

### 6.3. Reguła Bellmana

Niech L_t(q) oznacza przychód netto z wykonalnej sprzedaży q udziałów po bidzie. Dla stanu x_t obejmującego majątek i pozycję:

V_t(x_t) = max{ U(W_t po sprzedaży), E[V_(t+1)(x_(t+1)) | x_t, trzymaj] }.

Na końcu V_H odpowiada użyteczności rzeczywistej wypłaty netto z rozliczenia. Domyślna użyteczność to log dodatniego majątku, spójna z celem wzrostu; zero kapitału traktować jako ruinę, nie maskować dowolnym epsilonem w rankingu.

W izolowanej ocenie wyjść przyjąć, że po sprzedaży środki pozostają w gotówce, a pozostałe ekspozycje nie są zmieniane przez tę decyzję. To umożliwia rozdzielenie efektu samego wyjścia. W pełnym portfelu sprzedaż może finansować dalsze wejścia; tę korzyść ocenić oddzielnie w chronologicznym symulatorze.

Gdy brak prawidłowego bidu lub ilość na bidzie nie wystarcza na pełną sprzedaż, akcja sprzedaży jest niedostępna. Nie wypełniać braków przyszłą ceną.

### 6.4. Pierwsza implementacja

Zastosować regresyjne programowanie dynamiczne wstecz, inspirowane least-squares Monte Carlo:

1. Ustalić wspólną siatkę decyzji, początkowo co 5 s od wejścia, z jawnym końcem notowań i terminalnym rozliczeniem.
2. Każdy punkt siatki odtwarzać wyłącznie ze zdarzeń odebranych do tego punktu. Siatka jest częstotliwością decyzji wyjścia, nie opóźnieniem wejścia.
3. Na treningowych trajektoriach cofać się od końca, ucząc warunkową wartość kontynuacji.
4. Zacząć od regresji liniowej/ridge; porównać mały model drzewiasty, jeśli prosty wariant nie opisuje zależności.
5. Stosować podział/cross-fitting po całych rynkach i blokach czasu. Nigdy nie dzielić ticków jednego rynku między trening i ocenę.
6. Na outer test wykonywać tylko zamrożoną regułę. Przyszła ścieżka służy do rozliczenia, nie do wyboru momentu sprzedaży.
7. Gęstszą siatkę oceniać dopiero jako jawny, wspólny wariant rozdzielczości, jeśli dane ją wspierają.

Stan początkowy: czas do końca, p z wejścia, aktualny bid/ask/spread, zmiana ceny od wejścia, dostępna ilość, wielkość pozycji względem kapitału. Dodatkowe cechy BTC wyłącznie gdy są przyczynowo dostępne.

Kontrole: hold-to-resolution; prosta reguła sprzedaży przy zaniku przewagi po kosztach; wyuczona reguła kontynuacji. Dla prostej reguły jawnie wskazać, czy używa zamrożonego, czy aktualizowanego p.

Deep Optimal Stopping jest uzasadnieniem rodziny metod i ewentualnym późniejszym challengerem. Pierwsza implementacja nie wymaga sieci neuronowej.

## 7. Wspólny protokół porównania

### 7.1. Czas i niezależność

- Zachować historyczne bloki jako porównanie rozwojowe; okres był wcześniej używany do selekcji i nie staje się niezależnym holdoutem po zmianie polityki.
- Wszystkie estymatory, bootstrap, scenariusze, regresje kontynuacji i hiperparametry dopasowywać wyłącznie wewnątrz przeszłej części foldu.
- Purge opierać na końcu etykiety/trajektorii i dostępności wyniku, a nie tylko czasie wejścia.
- Zamrozić zakresy strojenia, seed, budżet i regułę wyboru przed outer evaluation. Nie powiększać przestrzeni po zobaczeniu wyniku testowego.
- Nowy, faktycznie nieużywany okres oceniać po zamrożeniu całej procedury. Nie blokuje to wcześniejszej implementacji ani badania rozwojowego.

### 7.2. Oddzielenie efektów

Przeprowadzić cztery porównania:

1. **Pełna polityka:** własne decyzje i stawki, wspólne rynki oraz model wykonania.
2. **Selekcja:** zamrożone reguły wyboru strony i wejścia, wspólna stawka $5 tam, gdzie wykonalna.
3. **Sizing:** zamrożona lista market/side/time; porównanie kwot na wspólnie wykonalnych transakcjach, bez ponownego wyboru strony. Pokazać też odrzucenia wywołane zmianą kwoty i ich wpływ w pełnym portfelu.
4. **Wyjścia:** identyczne wejścia i kwoty, różne reguły zamknięcia; osobno pełny portfel z reinwestowaniem uwolnionej gotówki.

Analiza wspólnego podzbioru nie zastępuje portfela z ograniczoną gotówką. Każda tabela podaje populację, liczbę transakcji i powody wykluczeń.

### 7.3. Cel i raport

Podstawowy cel selekcji: średni dzienny log-growth na wewnętrznej walidacji, liczony na wspólnej siatce dni. Ograniczenia ryzyka stanowią część definicji polityki. Raportować również front wzrost–ryzyko; nie zmieniać celu po outer wynikach.

Wymagane wyniki:

- końcowy kapitał i PnL, obrót, opłaty, liczba transakcji;
- dzienny log-growth i wyniki każdego bloku;
- drawdown kosztowy, likwidacyjny z coverage oraz maksymalna ekspozycja;
- czas pod wodą, wykorzystanie gotówki, minima/ilość/gotówka jako powody odrzuceń;
- sparowane różnice między politykami i niepewność z bloków czasu, z jawnym zastrzeżeniem zależności portfela;
- wpływ założeń wykonania przy małych kwotach; bez utożsamiania replayu z rzeczywistymi fillami;
- czas obliczeń, szczyt pamięci, zakres danych, cache hits i liczba zakończonych prób.

Brak stabilnego zwycięzcy jest dopuszczalnym wynikiem. Raport musi rozróżniać: słabą metodę w danym badaniu, niepewny wynik i brak danych do konkretnej metody.

## 8. Kolejność implementacji i warunki zakończenia

1. Zidentyfikować dostępne wejściowe cache i trajektorie, bez uruchamiania pełnego replayu tylko po to, by odczytać schemat.
2. Ujednolicić kontrakt akcji i ledger oraz przygotować kontrole izolujące selekcję/sizing. Zachować odtwarzalność wcześniejszego baseline'u.
3. Wdrożyć DRK wraz z kontrolą punktowej kalibracji.
4. Wdrożyć RCK i jawnie oznaczyć lokalną adaptację portfelową oraz granice gwarancji.
5. Przygotować wspólny cache trajektorii i wdrożyć OS. Brak trajektorii nie blokuje DRK/RCK; raport ma wskazać konkretny zakres brakujących danych.
6. Wykonać zamrożone porównanie, sporządzić wyniki i nieaktywny artefakt wybranej polityki albo uzasadniony brak wyboru.

Przed kosztownym przebiegiem wykonać reprezentatywny pomiar i podać szacowany zakres pracy. Współdzielić dane i scenariusze między próbami; checkpointować po foldach/próbach. Stosować [performance.md](../performance.md). Konfiguracja przez JSON/stałe, bez argumentów CLI.

Testy celowane: redukcja DRK do zwykłego Kelly'ego przy zerowej niepewności; analityczny przypadek binarny; minima i zaokrąglenia; brak kredytu; ryzyko przy skorelowanych pozycjach; brak przyszłych zdarzeń w wyjściach; niedostępność sprzedaży bez ilości; rachunek opłat sprzedaży i brak podwójnego settlementu; odtworzenie baseline'u.

Artefakty przyszłej implementacji: zamrożona konfiguracja z budżetem, manifest danych/modeli, tabela prób, log decyzji i transakcji, wyniki foldów oraz opis ograniczeń. Dokumentacja nie jest zleceniem uruchomienia treningu, replayu, handlu ani aktywacji modelu w tym commicie.

## 9. Literatura i zakres wykorzystania

1. **Sun, Boyd — Distributional Robust Kelly Gambling.** Zbiór niepewnych rozkładów i maksymalizacja najgorszego oczekiwanego log-growth. [Publikacja i kod](https://web.stanford.edu/~boyd/papers/robust_kelly.html).
2. **Busseti, Ryu, Boyd — Risk-Constrained Kelly Gambling (2016).** Ograniczenie ryzyka zejścia majątku poniżej ustalonego poziomu i wypukłe ograniczenie potęgowe. [Publikacja i kod](https://stanford.edu/~boyd/papers/kelly.html).
3. **Longstaff, Schwartz — Valuing American Options by Simulation: A Simple Least-Squares Approach (2001).** Regresyjna estymacja wartości kontynuacji. [DOI](https://doi.org/10.1093/rfs/14.1.113).
4. **Becker, Cheridito, Jentzen — Deep Optimal Stopping (2019).** Uczenie reguły zatrzymania z trajektorii; bardziej złożony wariant rodziny OS. [JMLR](https://jmlr.org/papers/v20/18-232.html).

Kalibracja warunkowa, sposób scenariuszy portfela, ograniczenie do małych zleceń i siatka wyjść są proponowanymi adaptacjami dla tego repozytorium, a nie twierdzeniami o wynikach tych publikacji na Polymarket.
