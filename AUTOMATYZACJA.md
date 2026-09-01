# Automatyzacja Mini Lotto

Repozytorium uruchamia analizę codziennie o 23:30 czasu polskiego.

- Od poniedziałku do soboty używany jest zapisany model (`daily`).
- W niedzielę wykonywany jest pełny trening i walk-forward (`train`).
- Jeśli model nie istnieje, pierwsze uruchomienie automatycznie wybiera trening.
- Aktualne wyniki są pobierane i walidowane przed uruchomieniem modelu.
- Najnowszy raport trafia do `reports/latest.md`, a kopia do `reports/archive/`.
- Poprzednie dwa zestawy są przechowywane w `.automation/last_prediction.json`.
- Model PyTorch jest przechowywany w pamięci podręcznej GitHub Actions.

## Uruchomienie ręczne

Na GitHubie otwórz `Actions`, wybierz `Mini Lotto - analiza i predykcja`,
kliknij `Run workflow` i wybierz `auto`, `daily` albo `train`.

## Work i Dysk Google

Zaplanowane zadanie Work powinno odczytywać `reports/latest.md`, przedstawiać
podsumowanie użytkownikowi i zapisywać kopię raportu w folderze `MiniLotto`
na Dysku Google. To zadanie można utworzyć dopiero po połączeniu Google Drive
z Work i potwierdzeniu dostępu odczytem testowym.
