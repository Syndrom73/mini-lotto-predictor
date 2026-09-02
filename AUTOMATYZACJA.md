# Automatyzacja Mini Lotto

Repozytorium uruchamia analizę codziennie o 22:30 czasu polskiego.

- Od poniedziałku do soboty używany jest zapisany model (`daily`).
- W niedzielę wykonywany jest pełny trening i walk-forward (`train`).
- Jeśli model nie istnieje, pierwsze uruchomienie automatycznie wybiera trening.
- Aktualne wyniki są pobierane i walidowane przed uruchomieniem modelu.
- Najnowszy raport trafia do `reports/latest.md`, a kopia do `reports/archive/`.
- Poprzednie dwa zestawy są przechowywane w `.automation/last_prediction.json`.
- Historia prognoz i późniejszych wyników ich kontroli trafia do
  `.automation/prediction_history.csv`.
- Model PyTorch jest przechowywany w pamięci podręcznej GitHub Actions.

## Modele i kontrola jakości

Końcowy wynik łączy model statystyczny z bazową siecią MLP. Dodatkowy model
hybrydowy analizuje ostatnie losowania za pomocą temporalnej sieci CNN, a
pozostałe cechy za pomocą MLP. CNN zostaje dopuszczony do końcowej predykcji
tylko wtedy, gdy poprawia Brier score na zbiorze walidacyjnym. Pełny trening
obu sieci i walk-forward odbywa się raz w tygodniu; w pozostałe dni używany
jest zapisany bundle.

## Uruchomienie ręczne

Na GitHubie otwórz `Actions`, wybierz `Mini Lotto - analiza i predykcja`,
kliknij `Run workflow` i wybierz `auto`, `daily` albo `train`.

## Work i Dysk Google

Zaplanowane zadanie Work powinno odczytywać `reports/latest.md`, przedstawiać
podsumowanie użytkownikowi i zapisywać kopię raportu w folderze `MiniLotto`
na Dysku Google. To zadanie można utworzyć dopiero po połączeniu Google Drive
z Work i potwierdzeniu dostępu odczytem testowym.
