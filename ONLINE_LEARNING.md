# Douczanie modelu Mini Lotto

Tryb daily ocenia prognozy zapisane przed wynikiem, aktualizuje obie sieci
na nowych wynikach i zapisuje checkpoint. Każdy nowy wynik jest przetwarzany
chronologicznie: dwie aktualizacje AdamW z learning rate 0.00003 i maksymalnie
127 wcześniejszymi przykładami. Replay celowo powtarza starsze przykłady.
BatchNorm i dropout pozostają w trybie eval; gradienty nadal są obliczane.
learned_through zapobiega powtórzeniu aktualizacji po ponownym uruchomieniu.

Po pełnym treningu metryki testowe zostają obliczone przed douczaniem.
Następnie model produkcyjny przetwarza także część walidacyjną i testową.
Zapisane historyczne metryki nie opisują tego późniejszego modelu.
Kalibratory i udziały modeli pozostają stałe do następnego pełnego treningu;
ich dopasowanie po zmianie wag wymaga obserwacji wyników produkcyjnych.

probability_history.json zawiera 42 prawdopodobieństwa, rzeczywiste dwa
zestawy, datę utworzenia, numer danych źródłowych i wersję modelu.
Audyt mierzy Brier i trafienia rzeczywistych zestawów. Starszych prognoz
bez zapisanych prawdopodobieństw nie rekonstruujemy po poznaniu wyniku.
Nie jest to pełny historyczny backtest selektora dwóch zestawów; jest to
ocena kolejnych rzeczywistych prognoz od wdrożenia.

GitHub zapisuje model w cache po każdym uruchomieniu z nowymi danymi oraz
w artefakcie wraz z raportem (90 dni). Brak cache powoduje pełny trening
w trybie auto. Wyraźny tryb daily przy braku modelu kończy się błędem.
Pierwsza migracja starego modelu doucza pominięte wcześniej 800 wyników
i ewentualne nowsze losowania, więc może potrwać dłużej niż zwykły dzień.

Testy: python -m unittest discover -s tests -p 'test_online_learning.py' -v
Testy używają danych syntetycznych i nie stanowią dowodu skuteczności gry.
Wyniki losowań są losowe; douczanie nie gwarantuje wygranej.


## Feedback i trening co dwa dni — 2026-09-10

Douczanie używa BCE oraz dodatkowego różniczkowalnego błędu Brier sieci.
Przed Brier korygowany jest mnożnik szans wynikający z ważonego BCE.
To pomocniczy Brier sieci, nie Brier skalibrowanej końcowej mieszanki z raportu.
Oceny uprzednio zapisanych prognoz zwiększają wagę danego przykładu maksymalnie
z 1 do 1.5: do 0.25 za Brier gorszy od 5/42 * 37/42 i do 0.25 za średnią
trafień obu zestawów poniżej 25/42. Brak oceny daje wagę 1.
Jest to eksperymentalna heurystyka, nie potwierdzona poprawa trafności.
Używane są wyłącznie ocenione prognozy sporządzone przed danym losowaniem;
replay obejmuje wyłącznie przykłady wcześniejsze od aktualizowanego celu.
Pełny trening zachowuje niezależny test; feedback jest stosowany w późniejszym
produkcyjnym douczaniu. Architektura i liczba kroków nie zmieniają się automatycznie.

Tryb auto wykonuje pełny trening, gdy od ostatniego udanego pełnego treningu
minęły co najmniej dwa dni kalendarzowe w Europe/Warsaw. Brak znacznika uruchamia
pierwszy pełny trening przy najbliższym nowym wyniku (plan: wieczór 10 września).
Data jest zapisywana dopiero po sukcesie w .automation/last_full_training.txt.
W pozostałe dni działa douczanie. Bez nowego wyniku model nie jest uruchamiany.
Harmonogram działa również na granicy miesięcy; ręczny tryb train/daily ma pierwszeństwo.
Ocena skuteczności musi korzystać z kolejnych prognoz zapisanych przed wynikami,
nie ze spadku straty na danych użytych już do douczania.
