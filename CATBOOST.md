# CatBoost Ranker — nowy silnik Mini Lotto

Nowy domyślny backend to `catboost`. Dotychczasowy model pozostaje dostępny
przez `--backend legacy`, aby umożliwić porównanie i odwrócenie migracji.
Nie korzystamy z wag MLP/CNN do inicjalizacji CatBoost.

## Uczenie i kara

Każdy przykład to liczba 1–42, a grupa to losowanie. Cechy obejmują numer
liczby, częstości z 5/10/25/50/100/250 wcześniejszych losowań, przerwę,
zmianę częstości oraz wystąpienie w poprzednim wyniku. Wynik losowania i
przyszłe wyniki nigdy nie wchodzą do jego cech. Historyczny okres przed
losowaniem 129 jest wyłączony z powodu innego zakresu liczb.

Porównujemy `PairLogit` (kara za odwróconą kolejność trafionej i nietrafionej
liczby) oraz `YetiRank:mode=NDCG;top=5` (nacisk na początek rankingu).
Oba mają ten sam limit 160 drzew, głębokość 4, L2=8, seed=42, dwa wątki CPU.
Parametry nie są zmieniane po zobaczeniu testu. Nie zwiększamy arbitralnie
wagi ostatniej porażki: może ona być zwykłą losowością.

Ostatnie 200 losowań przed granicą selekcji dzielimy na 100 do kalibracji
logistycznej i 100 do wyboru funkcji kary i selektora. Drzewa widzą tylko
wcześniejszą historię (po rozgrzewce 250 losowań). Kryterium wyboru:
częstość >=3/5 w którymkolwiek zestawie, następnie suma średnich trafień,
następnie Brier. Cztery kombinacje, bez rozległego strojenia.

Ujemny współczynnik kalibracji powoduje użycie równych prawdopodobieństw.
Marginalne estymaty są normalizowane do sumy 5; nie są gwarancją ani modelem
łącznego rozkładu pięciu kul.

## Dwa zestawy

`greedy` przydziela najwyższe dostępne oceny najpierw do zestawu 1,
`balanced` przydziela je na przemian. Oba dają dwa rozłączne zestawy po pięć
liczb, maksymalnie trzy wspólne liczby z każdym poprzednim zestawem.
Remisy rozstrzyga seed zależny od numeru losowania. Nie przeglądamy wszystkich
850668 kombinacji. Wybór strategii jest empiryczny; nie twierdzimy, że
maksymalizuje dokładne prawdopodobieństwo >=3/5.

## Codzienne działanie i zgodność

Dotychczasowe argumenty CLI, nazwy sekcji raportu, `reports/latest.md`,
archiwum, pliki stanu i historia prognoz pozostają zachowane. Raport podaje
rzeczywistą granicę uczenia, kalibracji i najnowszych cech. Nie zawiera
nieaktualnego opisu udziałów MLP/CNN.

Model zapisujemy atomowo jako `.automation/mini_lotto_catboost.zip`:
natywny plik CatBoost + metadane JSON, bez pickle. Cache ma osobny prefiks.
`daily` wczytuje model i aktualizuje cechy; **nie doucza drzew po pojedynczym
losowaniu**. Pełne uczenie zachowuje obecny harmonogram repozytorium
(co dwa dni), godziny pobierania wyników i ponowienia. Nie zmieniamy go
na tygodniowy. Kalibracja i walidacja pozostają poza uczeniem drzew także
w modelu produkcyjnym, więc raport nie oznacza ich jako danych treningowych.

Prognoza oczekująca na wynik nie jest nadpisywana. Pełne uczenie może
odświeżyć model, ale nie zmienia oczekującej prognozy. Zmiana lub cofnięcie
historii względem zapisanej sumy kontrolnej wymaga ponownego treningu.
Przy pierwszym użyciu nowego backendu brak jego cache uruchamia trening
w trybie auto. Wybranie jawnego daily bez modelu kończy się błędem.

Przykład (używaj ścieżek tymczasowych do prób):

```bash
python mini_lotto_predictor.py --backend catboost --mode train \
  --csv-path wyniki-minilotto.csv \
  --bundle-path .automation/mini_lotto_catboost.zip \
  --state-path .automation/last_prediction.json \
  --report-path reports/latest.md \
  --prediction-history-path .automation/prediction_history.csv
```

## Niezależny test historyczny

```bash
python ranker_backtest.py --csv-path wyniki-minilotto.csv \
  --output reports/experiments/catboost-holdout.json --legacy
python -m unittest discover -s tests -v
```

Ostatnie 400 losowań jest testem. Wcześniejsze 200 służy kalibracji i selekcji.
Podczas testu model jest zamrożony, cechy aktualizowane wyłącznie po kolejnych
wynikach. Selekcja zestawów ma własną historię rotacji dla każdej metody.
Zapis obejmuje każdy typ, trafienia, histogramy 0–5, Brier, przedział Wilsona,
czas, granice czasowe, ustawienia i SHA256 danych.

Porównania: CatBoost, częstości z poprzednich 250 losowań, wybór losowy z tymi
samymi ograniczeniami i 200 powtórzeń losowego punktu odniesienia. Opcjonalny
`--legacy` uczy stare sieci ponownie wyłącznie przed testem i porównuje ich
prawdopodobieństwa **z nowym selektorem**. To porównanie silników oceniania,
a nie pełnej starej polityki zestawów. Stare modele zachowują własny większy
budżet uczenia; czas jest raportowany osobno.

Test jest niezależny od strojenia tej implementacji, ale retrospektywny:
te same losowania mogły być oglądane we wcześniejszych pracach projektu.
Nie należy przedstawiać go jako nowego prospektywnego dowodu przewagi.
Przedział Wilsona jest opisowy, nie uwzględnia możliwej zależności między
prognozami. Po oglądaniu wyniku nie stroimy modelu na tym samym teście.

Workflow `CatBoost validation` uruchamia testy na PR. Ręcznie można dołączyć
benchmark historyczny; ma tylko prawo odczytu i zapisuje artefakt, bez
zmiany raportu produkcyjnego. Harmonogram GitHub uruchamia się z gałęzi
domyślnej: osobna gałąź nie zastępuje produkcji przed scaleniem.

Gra jest losowa; zmiana algorytmu nie gwarantuje przewagi ani wygranej.
