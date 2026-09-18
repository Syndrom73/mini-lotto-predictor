# CatBoost: test historyczny, 17 września 2026

400 losowań: **7039–7438**, od **2025-08-07 do 2026-09-10**.
Snapshot pochodzi z historycznego CSV MultiPasko; nie obejmuje późniejszych
losowań. Pełne prognozy, trafienia, granice podziału i SHA256:
[catboost-holdout.json](catboost-holdout.json).

| Metoda | Losowania z >=3/5 w jednym z dwóch zestawów | Odsetek | Średnie trafienia Z1 / Z2 | Brier |
|---|---:|---:|---:|---:|
| CatBoost Ranker | 6 / 400 | 1,50% | 0,610 / 0,640 | 0,104878 |
| Częstości z poprzednich 250 losowań | 7 / 400 | 1,75% | 0,635 / 0,543 | 0,105270 |
| Losowy wybór, jeden przebieg | 6 / 400 | 1,50% | 0,575 / 0,620 | 0,104875 |
| Stare MLP/CNN + nowy selektor | 15 / 400 | 3,75% | 0,650 / 0,603 | 0,104865 |

Wybrana na wcześniejszej walidacji konfiguracja: PairLogit, 160 drzew,
głębokość 4, selektor balanced. Test nie uczestniczył w wyborze. Wszystkie
trafienia >=3 w tej tabeli to dokładnie 3/5; nie było 4/5 ani 5/5.

Dla CatBoost opisowy przedział Wilsona 95% wynosi 0,689–3,233%.
W 200 losowych przebiegach środkowe 95% odsetków wyniosło 0,744–3,250%,
mediana 1,50%. Dokładny punkt odniesienia przy niezależnych losowaniach i
dwóch rozłącznych zestawach wynosi 1,6096% na losowanie.

**Ten test nie potwierdził przewagi CatBoost.** Stary silnik z nowym
selektorem uzyskał więcej trafień. Nie jest to porównanie ze starą pełną
polityką tworzenia zestawów ani dowód trwałej przewagi starego modelu.
Stare sieci miały własny budżet i wewnętrzny podział treningowy.

Cały benchmark trwał około 63,5 s w tym środowisku; uczenie i predykcja
starych sieci zajęły około 25,4 s. Nie porównujemy tych czasów jako
równego budżetu: pozostały czas obejmuje dwa modele CatBoost, selekcję,
cechy i losowe symulacje. Nie deklarujemy przyspieszenia całego procesu.

To retrospektywna ocena zamrożonych modeli, z przyrostowym dostępem do
wcześniejszych wyników w cechach. Okres mógł być analizowany we wcześniejszych
pracach projektu. Nie zastępuje przyszłego, zapisanego przed losowaniami testu.
Po uzyskaniu tych wyników nie zmieniano parametrów dla poprawienia wyniku.

Odtworzenie na nowszym CSV:

```bash
python ranker_backtest.py --csv-path wyniki-minilotto.csv --through-draw 7438 \
  --test-size 400 --validation-size 200 --iterations 160 --legacy \
  --output /tmp/catboost-holdout.json
```
