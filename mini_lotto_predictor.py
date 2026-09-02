"""
MINI LOTTO PREDICTOR — WERSJA POPRAWIONA

Model:
  * historia współczesnych losowań Mini Lotto (5 z 42),
  * cechy wyliczane wyłącznie z losowań wcześniejszych,
  * MLP w PyTorch,
  * statystyczny model bazowy,
  * kalibracja Platta na surowych logitach NN,
  * temporalny train/validation/test,
  * opcjonalny expanding-window walk-forward,
  * dokładne przeszukanie wszystkich C(42, 5) = 850 668 zestawów,
  * dwa zestawy z kontrolowanym maksymalnym overlapem,
  * kompletny zapis i odczyt modelu wraz z historią i scalerem.

Ważne:
  Wyniki losowań są losowe. Kod porządkuje analizę i umożliwia uczciwy
  backtest, ale nie zwiększa matematycznego prawdopodobieństwa pojedynczego
  zakładu ponad prawdopodobieństwo właściwe grze 5 z 42.

Wymagania:
  pip install numpy pandas scipy scikit-learn torch

Google Colab:
  1. Wgraj plik wyniki-minilotto.csv do /content/.
  2. Zamontuj Dysk Google, aby zachować model i poprzednią prognozę.
  3. Opcjonalnie ustaw CSV_PATH i NEXT_DRAW_DATE na końcu pliku.
  4. Uruchom cały kod.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import warnings
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import StringIO
from itertools import combinations
from math import comb
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 0. KONFIGURACJA
# ============================================================

SEED = 20260901


@dataclass
class Config:
    # Mini Lotto
    n_numbers: int = 42
    draw_size: int = 5
    modern_start_draw: int = 129

    # Cechy
    sequence_length: int = 20
    rolling_windows: Tuple[int, ...] = (5, 10, 20, 50, 100, 250)
    pair_window: int = 250

    # Podział temporalny
    validation_size: int = 400
    test_size: int = 400

    # Sieć
    hidden_1: int = 384
    hidden_2: int = 192
    hidden_3: int = 96
    dropout: float = 0.30
    learning_rate: float = 3e-4
    weight_decay: float = 2e-5
    epochs: int = 120
    batch_size: int = 128
    patience: int = 15

    # Ensemble
    default_nn_weight: float = 0.60
    hybrid_max_weight: float = 0.50
    hybrid_min_brier_improvement: float = 1e-5

    # Temporalny CNN + MLP. CNN widzi macierz: czas x 42 liczby,
    # a MLP pozostałe cechy statystyczne.
    cnn_channels_1: int = 32
    cnn_channels_2: int = 64
    hybrid_context_hidden: int = 128

    # Dokładne wyszukiwanie zestawów
    individual_score_weight: float = 1.0
    pair_score_weight: float = 0.20
    spread_weight: float = 0.04
    max_overlap: int = 0
    # Każdy nowy zestaw musi zmienić co najmniej 2 liczby względem
    # każdego zestawu typowanego na poprzednie losowanie.
    max_previous_overlap: int = 3

    # Prawdziwy expanding-window walk-forward.
    # Jest kosztowny, ponieważ każdy fold trenuje nową sieć.
    walk_forward_enabled: bool = True
    walk_forward_folds: int = 3
    walk_forward_block_size: int = 100
    walk_forward_validation_size: int = 250
    walk_forward_epochs: int = 60
    walk_forward_patience: int = 10

    # Środowisko
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()
REQUIRED_COLUMNS = [
    "Numer", "Dzien", "Miesiac", "Rok", "L1", "L2", "L3", "L4", "L5"
]
NUMBER_COLUMNS = ["L1", "L2", "L3", "L4", "L5"]
FEATURE_VERSION = 3


# ============================================================
# 1. POWTARZALNOŚĆ I PODSTAWOWE FUNKCJE
# ============================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Pełna deterministyczność może nieznacznie spowolnić GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def numbers_to_multihot(numbers: Sequence[int], n_numbers: int = 42) -> np.ndarray:
    out = np.zeros(n_numbers, dtype=np.float32)
    for number in numbers:
        if not 1 <= int(number) <= n_numbers:
            raise ValueError(f"Liczba {number} jest poza zakresem 1..{n_numbers}.")
        out[int(number) - 1] = 1.0
    return out


def top_numbers(scores: np.ndarray, top_k: int = 5) -> set[int]:
    indexes = np.argsort(np.asarray(scores))[-top_k:]
    return set((indexes + 1).tolist())


def normalize_expected_count(values: np.ndarray, expected_count: int = 5) -> np.ndarray:
    """Skaluje nieujemne wartości do [0, 1] tak, aby ich suma wynosiła 5."""
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    values = np.clip(values, 1e-12, None)

    # Szukamy mnożnika alpha dla sum(min(alpha*x, 1)) = expected_count.
    low, high = 0.0, 1.0
    while np.minimum(high * values, 1.0).sum() < expected_count:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if np.minimum(middle * values, 1.0).sum() < expected_count:
            low = middle
        else:
            high = middle
    result = np.minimum(high * values, 1.0)
    return result.astype(np.float64)


# ============================================================
# 2. WCZYTANIE I WALIDACJA HISTORII
# ============================================================

def _read_csv_robust(path: str) -> pd.DataFrame:
    errors = []
    for separator in (";", ",", "\t"):
        try:
            candidate = pd.read_csv(path, sep=separator, encoding="utf-8-sig")
            candidate.columns = [str(column).strip() for column in candidate.columns]
            if set(REQUIRED_COLUMNS).issubset(candidate.columns):
                return candidate
        except Exception as exc:
            errors.append(f"sep={separator!r}: {exc}")
    raise ValueError(
        "Nie udało się odczytać wymaganych kolumn CSV. "
        "Oczekiwane kolumny: " + ", ".join(REQUIRED_COLUMNS) + ". "
        + " | ".join(errors)
    )


def load_history(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Nie znaleziono pliku historii: {path}")
    df = _read_csv_robust(path)
    for column in REQUIRED_COLUMNS:
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
        except Exception as exc:
            raise ValueError(f"Kolumna {column} zawiera wartość nienumeryczną: {exc}") from exc
        if df[column].isna().any():
            raise ValueError(f"Kolumna {column} zawiera brakujące wartości.")
        if not np.all(np.equal(df[column], np.floor(df[column]))):
            raise ValueError(f"Kolumna {column} zawiera wartości niecałkowite.")
        df[column] = df[column].astype(int)
    validate_history(df)
    return prepare_history(df)


def validate_history(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Brak wymaganych kolumn: {missing}")
    if df["Numer"].duplicated().any():
        duplicates = df.loc[df["Numer"].duplicated(), "Numer"].tolist()
        raise ValueError(f"Zduplikowane numery losowań: {duplicates[:10]}")

    sorted_numbers = np.sort(df["Numer"].to_numpy(dtype=int))
    if len(sorted_numbers) > 1 and not np.all(np.diff(sorted_numbers) == 1):
        gaps = sorted_numbers[1:][np.diff(sorted_numbers) != 1]
        raise ValueError(
            "Numeracja losowań nie jest ciągła. "
            f"Pierwsze numery po wykrytych lukach: {gaps[:10].tolist()}"
        )

    dates = pd.to_datetime(
        dict(year=df["Rok"], month=df["Miesiac"], day=df["Dzien"]),
        errors="coerce",
    )
    if dates.isna().any():
        bad_rows = df.loc[dates.isna(), ["Numer", "Dzien", "Miesiac", "Rok"]]
        raise ValueError(f"Niepoprawne daty:\n{bad_rows.head(10)}")

    for row in df.itertuples(index=False):
        draw_number = int(getattr(row, "Numer"))
        balls = [int(getattr(row, column)) for column in NUMBER_COLUMNS]
        if len(set(balls)) != CFG.draw_size:
            raise ValueError(f"Losowanie {draw_number}: liczby nie są unikalne: {balls}")
        if min(balls) < 1:
            raise ValueError(f"Losowanie {draw_number}: wykryto liczbę mniejszą od 1.")
        if draw_number >= CFG.modern_start_draw and max(balls) > CFG.n_numbers:
            raise ValueError(
                f"Losowanie {draw_number}: liczba poza współczesnym zakresem 1..42."
            )


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Date"] = pd.to_datetime(
        dict(year=result["Rok"], month=result["Miesiac"], day=result["Dzien"])
    )
    result = result.sort_values("Numer").reset_index(drop=True)
    if not result["Date"].is_monotonic_increasing:
        raise ValueError("Daty nie są rosnące zgodnie z numerami losowań.")
    return result


def get_modern_history(df: pd.DataFrame) -> pd.DataFrame:
    modern = df.loc[df["Numer"] >= CFG.modern_start_draw].copy().reset_index(drop=True)
    if modern.empty:
        raise ValueError(f"Brak losowań od numeru {CFG.modern_start_draw}.")
    return modern


def history_to_matrix(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for row in df.itertuples(index=False):
        rows.append(numbers_to_multihot([getattr(row, column) for column in NUMBER_COLUMNS]))
    return np.asarray(rows, dtype=np.float32)


# ============================================================
# 3. FEATURE ENGINEERING BEZ LEAKAGE
# ============================================================

def rolling_frequency(history: np.ndarray, current_idx: int, window: int) -> np.ndarray:
    sample = history[max(0, current_idx - window):current_idx]
    if len(sample) == 0:
        return np.zeros(CFG.n_numbers, dtype=np.float32)
    return sample.mean(axis=0).astype(np.float32)


def calculate_gaps(history: np.ndarray, current_idx: int) -> np.ndarray:
    previous = history[:current_idx]
    gaps = np.empty(CFG.n_numbers, dtype=np.float32)
    for index in range(CFG.n_numbers):
        appearances = np.flatnonzero(previous[:, index] > 0)
        gaps[index] = current_idx + 1 if len(appearances) == 0 else current_idx - appearances[-1]
    return gaps


def calculate_trend(history: np.ndarray, current_idx: int) -> np.ndarray:
    return rolling_frequency(history, current_idx, 10) - rolling_frequency(history, current_idx, 50)


def regularized_pair_matrix(history: np.ndarray, current_idx: int, window: int) -> np.ndarray:
    """Regularizowany score związku par, a nie sama surowa liczba wystąpień."""
    sample = history[max(0, current_idx - window):current_idx].astype(np.float64)
    if len(sample) < 2:
        return np.zeros((CFG.n_numbers, CFG.n_numbers), dtype=np.float64)

    count = sample.sum(axis=0)
    cooccurrence = sample.T @ sample
    observed = cooccurrence / len(sample)
    frequency = count / len(sample)
    expected = np.outer(frequency, frequency)

    # Lift z kurczeniem rzadkich obserwacji w stronę zera.
    lift = (observed + 1.0 / len(sample)) / (expected + 1.0 / len(sample))
    reliability = cooccurrence / (cooccurrence + 5.0)
    association = np.tanh(np.log(np.clip(lift, 1e-8, None))) * reliability
    np.fill_diagonal(association, 0.0)
    return association


def calculate_pair_context_strength(
    history: np.ndarray, current_idx: int, window: int
) -> np.ndarray:
    """Związek każdej liczby z liczbami z ostatniego dostępnego losowania."""
    if current_idx == 0:
        return np.zeros(CFG.n_numbers, dtype=np.float32)
    matrix = regularized_pair_matrix(history, current_idx, window)
    previous_draw = np.flatnonzero(history[current_idx - 1] > 0)
    if len(previous_draw) == 0:
        return np.zeros(CFG.n_numbers, dtype=np.float32)
    return matrix[:, previous_draw].mean(axis=1).astype(np.float32)


def sequential_draw_features(history: np.ndarray, current_idx: int, length: int) -> np.ndarray:
    sequence = history[max(0, current_idx - length):current_idx]
    if len(sequence) < length:
        padding = np.zeros((length - len(sequence), CFG.n_numbers), dtype=np.float32)
        sequence = np.vstack([padding, sequence])
    return sequence.flatten().astype(np.float32)


def calendar_features(date: pd.Timestamp) -> np.ndarray:
    day_of_week = int(date.dayofweek)
    month = int(date.month)
    return np.asarray(
        [
            math.sin(2 * math.pi * day_of_week / 7),
            math.cos(2 * math.pi * day_of_week / 7),
            math.sin(2 * math.pi * (month - 1) / 12),
            math.cos(2 * math.pi * (month - 1) / 12),
        ],
        dtype=np.float32,
    )


def build_feature_vector(
    df: pd.DataFrame,
    history: np.ndarray,
    current_idx: int,
    target_date: pd.Timestamp,
) -> np.ndarray:
    features = [sequential_draw_features(history, current_idx, CFG.sequence_length)]
    features.extend(rolling_frequency(history, current_idx, w) for w in CFG.rolling_windows)
    features.append(np.log1p(calculate_gaps(history, current_idx)))
    features.append(calculate_trend(history, current_idx))
    features.append(calculate_pair_context_strength(history, current_idx, CFG.pair_window))
    features.append(calendar_features(pd.Timestamp(target_date)))
    return np.concatenate(features).astype(np.float32)


def dataset_offset() -> int:
    return max(CFG.sequence_length, max(CFG.rolling_windows))


def sequence_feature_count() -> int:
    """Liczba początkowych cech należących do sekwencji losowań."""

    return CFG.sequence_length * CFG.n_numbers


def build_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    history = history_to_matrix(df)
    offset = dataset_offset()
    X, Y = [], []
    for index in range(offset, len(df)):
        X.append(build_feature_vector(df, history, index, df.iloc[index]["Date"]))
        Y.append(history[index])
    if not X:
        raise ValueError(f"Za mało historii. Potrzeba więcej niż {offset} losowań.")
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


# ============================================================
# 4. PODZIAŁ TEMPORALNY
# ============================================================

@dataclass
class TemporalData:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    scaler: StandardScaler
    val_start_dataset: int
    test_start_dataset: int


def temporal_split(X: np.ndarray, Y: np.ndarray) -> TemporalData:
    test_start = len(X) - CFG.test_size
    val_start = test_start - CFG.validation_size
    if val_start <= CFG.batch_size:
        raise ValueError(
            f"Za mało przykładów ({len(X)}) dla validation_size={CFG.validation_size} "
            f"i test_size={CFG.test_size}."
        )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:val_start]).astype(np.float32)
    X_val = scaler.transform(X[val_start:test_start]).astype(np.float32)
    X_test = scaler.transform(X[test_start:]).astype(np.float32)
    return TemporalData(
        X_train, Y[:val_start], X_val, Y[val_start:test_start],
        X_test, Y[test_start:], scaler, val_start, test_start
    )


# ============================================================
# 5. SIEĆ I TRENING
# ============================================================

class MiniLottoMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, CFG.hidden_1),
            nn.BatchNorm1d(CFG.hidden_1),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.hidden_1, CFG.hidden_2),
            nn.BatchNorm1d(CFG.hidden_2),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.hidden_2, CFG.hidden_3),
            nn.GELU(),
            nn.Dropout(CFG.dropout / 2),
            nn.Linear(CFG.hidden_3, CFG.n_numbers),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MiniLottoTemporalHybrid(nn.Module):
    """Temporalny CNN dla historii oraz MLP dla cech statystycznych."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.sequence_features = sequence_feature_count()
        context_dim = self.input_dim - self.sequence_features
        if context_dim <= 0:
            raise ValueError("Brak cech kontekstowych dla modelu hybrydowego.")

        # Wejście CNN po reshape ma format [batch, liczby, czas]. Filtr
        # przesuwa się wyłącznie po kolejnych losowaniach, a nie po przypadkowo
        # ułożonych cechach statystycznych.
        self.temporal = nn.Sequential(
            nn.Conv1d(CFG.n_numbers, CFG.cnn_channels_1, kernel_size=3, padding=1),
            nn.BatchNorm1d(CFG.cnn_channels_1),
            nn.GELU(),
            nn.Conv1d(
                CFG.cnn_channels_1,
                CFG.cnn_channels_2,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(CFG.cnn_channels_2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.context = nn.Sequential(
            nn.Linear(context_dim, CFG.hybrid_context_hidden),
            nn.BatchNorm1d(CFG.hybrid_context_hidden),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.hybrid_context_hidden, CFG.hidden_3),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(CFG.cnn_channels_2 + CFG.hidden_3, CFG.hidden_2),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.hidden_2, CFG.n_numbers),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x[:, :self.sequence_features].reshape(
            -1, CFG.sequence_length, CFG.n_numbers
        )
        sequence = sequence.transpose(1, 2)
        context = x[:, self.sequence_features:]
        return self.head(torch.cat((self.temporal(sequence), self.context(context)), dim=1))


class WeightedBCE(nn.Module):
    def __init__(self):
        super().__init__()
        ratio = (CFG.n_numbers - CFG.draw_size) / CFG.draw_size
        weight = torch.full((CFG.n_numbers,), ratio, device=CFG.device)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(prediction, target)


@dataclass
class TrainingResult:
    model: nn.Module
    best_val_loss: float
    epochs_run: int


def train_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
    architecture: str = "mlp",
) -> TrainingResult:
    epochs = CFG.epochs if epochs is None else epochs
    patience = CFG.patience if patience is None else patience
    if architecture == "mlp":
        model: nn.Module = MiniLottoMLP(X_train.shape[1])
    elif architecture == "hybrid":
        model = MiniLottoTemporalHybrid(X_train.shape[1])
    else:
        raise ValueError(f"Nieznana architektura: {architecture}")
    model = model.to(CFG.device)
    criterion = WeightedBCE()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )
    dataset = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(Y_train, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset, batch_size=CFG.batch_size, shuffle=True, generator=generator,
        drop_last=False,
    )
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=CFG.device)
    Yv = torch.as_tensor(Y_val, dtype=torch.float32, device=CFG.device)

    best_state: Optional[dict] = None
    best_loss = float("inf")
    counter = 0
    epochs_run = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            # BatchNorm nie obsługuje jednoelementowej partii treningowej.
            if len(xb) < 2:
                continue
            xb, yb = xb.to(CFG.device), yb.to(CFG.device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError("Niefinitywna strata treningowa.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            value = criterion(model(Xv), Yv).item()
        if not math.isfinite(value):
            raise FloatingPointError("Niefinitywna strata walidacyjna.")
        epochs_run = epoch + 1
        if value < best_loss - 1e-5:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            break

    if best_state is None:
        raise RuntimeError("Nie udało się zapisać poprawnego stanu modelu.")
    model.load_state_dict(best_state)
    return TrainingResult(model, best_loss, epochs_run)


def model_logits(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(X, dtype=torch.float32, device=CFG.device)
    with torch.no_grad():
        return model(tensor).cpu().numpy()


# ============================================================
# 6. KALIBRACJA I MODEL STATYSTYCZNY
# ============================================================

def fit_platt_calibrator(logits: np.ndarray, targets: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(logits.reshape(-1, 1), targets.reshape(-1).astype(int))
    return calibrator


def calibrated_nn_probabilities(
    calibrator: LogisticRegression, logits: np.ndarray
) -> np.ndarray:
    flat = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
    result = flat.reshape(logits.shape)
    return np.asarray(
        [normalize_expected_count(row, CFG.draw_size) for row in result], dtype=np.float64
    )


def statistical_scores(history: np.ndarray, index: int) -> np.ndarray:
    freq10 = rolling_frequency(history, index, 10)
    freq20 = rolling_frequency(history, index, 20)
    freq50 = rolling_frequency(history, index, 50)
    freq100 = rolling_frequency(history, index, 100)
    gaps = calculate_gaps(history, index)

    # Łagodna funkcja recency. Nie premiuje bez ograniczeń skrajnie długiej przerwy.
    recency = np.exp(-gaps / 18.0)
    long_gap = 1.0 - np.exp(-gaps / 35.0)
    trend = np.clip(freq10 - freq50, -0.5, 0.5)
    raw = (
        0.18 * freq10 + 0.20 * freq20 + 0.25 * freq50 + 0.27 * freq100
        + 0.04 * recency + 0.03 * long_gap + 0.03 * (trend + 0.5)
    )
    return normalize_expected_count(raw, CFG.draw_size)


def statistical_predictions(history: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.asarray([statistical_scores(history, i) for i in range(start, end)])


def ensemble_probabilities(
    nn_probabilities: np.ndarray, statistical_probabilities: np.ndarray, nn_weight: float
) -> np.ndarray:
    mixed = nn_weight * nn_probabilities + (1.0 - nn_weight) * statistical_probabilities
    if mixed.ndim == 1:
        return normalize_expected_count(mixed, CFG.draw_size)
    return np.asarray([normalize_expected_count(row, CFG.draw_size) for row in mixed])


# ============================================================
# 7. METRYKI I ANALITYCZNY BASELINE
# ============================================================

@dataclass
class EvaluationMetrics:
    mean_hits: float
    hit_rate_0: float
    hit_rate_1: float
    hit_rate_2: float
    hit_rate_3: float
    hit_rate_4: float
    hit_rate_5: float
    rate_ge_2: float
    rate_ge_3: float
    brier: float


def evaluate_scores(scores: np.ndarray, targets: np.ndarray) -> EvaluationMetrics:
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    counts = []
    for score, target in zip(scores, targets):
        actual = set((np.flatnonzero(target > 0) + 1).tolist())
        counts.append(len(top_numbers(score, CFG.draw_size) & actual))
    counts = np.asarray(counts, dtype=int)
    rates = [float(np.mean(counts == k)) for k in range(6)]
    return EvaluationMetrics(
        mean_hits=float(counts.mean()),
        hit_rate_0=rates[0], hit_rate_1=rates[1], hit_rate_2=rates[2],
        hit_rate_3=rates[3], hit_rate_4=rates[4], hit_rate_5=rates[5],
        rate_ge_2=float(np.mean(counts >= 2)),
        rate_ge_3=float(np.mean(counts >= 3)),
        brier=float(brier_score_loss(targets.ravel(), np.clip(scores, 0, 1).ravel())),
    )


def analytical_random_baseline() -> EvaluationMetrics:
    denominator = comb(CFG.n_numbers, CFG.draw_size)
    probabilities = []
    for hits_count in range(CFG.draw_size + 1):
        numerator = comb(CFG.draw_size, hits_count) * comb(
            CFG.n_numbers - CFG.draw_size, CFG.draw_size - hits_count
        )
        probabilities.append(numerator / denominator)
    random_probability = CFG.draw_size / CFG.n_numbers
    # E[(p-y)^2] dla stałego, prawidłowego prawdopodobieństwa brzegowego.
    brier = random_probability * (1.0 - random_probability)
    return EvaluationMetrics(
        mean_hits=CFG.draw_size * CFG.draw_size / CFG.n_numbers,
        hit_rate_0=probabilities[0], hit_rate_1=probabilities[1],
        hit_rate_2=probabilities[2], hit_rate_3=probabilities[3],
        hit_rate_4=probabilities[4], hit_rate_5=probabilities[5],
        rate_ge_2=sum(probabilities[2:]), rate_ge_3=sum(probabilities[3:]),
        brier=brier,
    )


def optimize_ensemble_weight(
    nn_predictions: np.ndarray,
    stat_predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    best_weight = CFG.default_nn_weight
    best_key = (-float("inf"), -float("inf"), -float("inf"), -float("inf"))
    for weight in np.linspace(0.0, 1.0, 21):
        predictions = ensemble_probabilities(nn_predictions, stat_predictions, float(weight))
        metrics = evaluate_scores(predictions, targets)
        # Średnia liczba trafień jest znacznie stabilniejsza niż bardzo rzadkie
        # >=3/5. Rzadkie zdarzenia są dopiero kryteriami rozstrzygającymi remisy.
        key = (
            metrics.mean_hits,
            metrics.rate_ge_2,
            metrics.rate_ge_3,
            -metrics.brier,
        )
        if key > best_key:
            best_key = key
            best_weight = float(weight)
    return best_weight


def blend_hybrid_probabilities(
    base_predictions: np.ndarray,
    hybrid_predictions: np.ndarray,
    hybrid_weight: float,
) -> np.ndarray:
    mixed = (
        (1.0 - hybrid_weight) * base_predictions
        + hybrid_weight * hybrid_predictions
    )
    if mixed.ndim == 1:
        return normalize_expected_count(mixed, CFG.draw_size)
    return np.asarray(
        [normalize_expected_count(row, CFG.draw_size) for row in mixed],
        dtype=np.float64,
    )


def optimize_hybrid_weight(
    base_predictions: np.ndarray,
    hybrid_predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Dopuszcza CNN tylko po mierzalnej poprawie Brier na walidacji."""

    base_metrics = evaluate_scores(base_predictions, targets)
    best_weight = 0.0
    best_metrics = base_metrics
    for weight in np.linspace(0.05, CFG.hybrid_max_weight, 10):
        predictions = blend_hybrid_probabilities(
            base_predictions, hybrid_predictions, float(weight)
        )
        metrics = evaluate_scores(predictions, targets)
        if (
            metrics.brier < best_metrics.brier
            or (
                math.isclose(metrics.brier, best_metrics.brier, abs_tol=1e-12)
                and metrics.mean_hits > best_metrics.mean_hits
            )
        ):
            best_weight = float(weight)
            best_metrics = metrics

    improvement = base_metrics.brier - best_metrics.brier
    if improvement < CFG.hybrid_min_brier_improvement:
        return 0.0
    return best_weight


def print_metrics(name: str, metrics: EvaluationMetrics) -> None:
    print("\n------------------------------")
    print(name)
    print("------------------------------")
    print(f"Mean hits: {metrics.mean_hits:.4f}")
    print(f">=2/5:     {100 * metrics.rate_ge_2:.3f}%")
    print(f">=3/5:     {100 * metrics.rate_ge_3:.3f}%")
    print(f"3/5:       {100 * metrics.hit_rate_3:.3f}%")
    print(f"4/5:       {100 * metrics.hit_rate_4:.5f}%")
    print(f"5/5:       {100 * metrics.hit_rate_5:.7f}%")
    print(f"Brier:     {metrics.brier:.6f}")


# ============================================================
# 8. DOKŁADNE WYSZUKIWANIE ZESTAWÓW — BEZ GA
# ============================================================

Individual = Tuple[int, ...]


def set_fitness(
    individual: Individual, probabilities: np.ndarray, pair_matrix: np.ndarray
) -> float:
    indexes = np.asarray(individual, dtype=int) - 1
    individual_part = float(probabilities[indexes].sum())
    pair_values = [pair_matrix[a, b] for a, b in combinations(indexes, 2)]
    pair_part = float(np.mean(pair_values)) if pair_values else 0.0
    spread_part = float((indexes.max() - indexes.min()) / (CFG.n_numbers - 1))
    return (
        CFG.individual_score_weight * individual_part
        + CFG.pair_score_weight * pair_part
        + CFG.spread_weight * spread_part
    )


def exact_best_set(
    probabilities: np.ndarray,
    pair_matrix: np.ndarray,
    forbidden_set: Optional[Individual] = None,
    max_overlap: Optional[int] = None,
    overlap_constraints: Optional[Sequence[Tuple[Individual, int]]] = None,
) -> Tuple[Individual, float]:
    max_overlap = CFG.max_overlap if max_overlap is None else max_overlap
    constraints: List[Tuple[set[int], int]] = []
    if forbidden_set:
        constraints.append((set(forbidden_set), max_overlap))
    for numbers, limit in overlap_constraints or ():
        if not 0 <= limit <= CFG.draw_size:
            raise ValueError("Limit wspólnych liczb musi należeć do zakresu 0..5.")
        constraints.append((set(numbers), int(limit)))

    best_set: Optional[Individual] = None
    best_fitness = -float("inf")
    for candidate in combinations(range(1, CFG.n_numbers + 1), CFG.draw_size):
        candidate_numbers = set(candidate)
        if any(
            len(candidate_numbers & blocked) > limit
            for blocked, limit in constraints
        ):
            continue
        value = set_fitness(candidate, probabilities, pair_matrix)
        if value > best_fitness:
            best_set = candidate
            best_fitness = value
    if best_set is None:
        raise RuntimeError("Nie znaleziono zestawu spełniającego ograniczenia overlapu.")
    return best_set, best_fitness


@dataclass
class Prediction:
    next_draw_number: int
    next_draw_date: pd.Timestamp
    set_1: Individual
    set_2: Individual
    set_1_fitness: float
    set_2_fitness: float
    ranking: List[Tuple[int, float]]
    hybrid_weight: float = 0.0


def generate_two_sets(
    final_probabilities: np.ndarray,
    history: np.ndarray,
    next_draw_number: int,
    next_draw_date: pd.Timestamp,
    previous_prediction: Optional["StoredPrediction"] = None,
) -> Prediction:
    pair_matrix = regularized_pair_matrix(history, len(history), CFG.pair_window)
    previous_constraints: List[Tuple[Individual, int]] = []
    if previous_prediction is not None:
        previous_constraints = [
            (previous_prediction.set_1, CFG.max_previous_overlap),
            (previous_prediction.set_2, CFG.max_previous_overlap),
        ]

    set_1, fitness_1 = exact_best_set(
        final_probabilities,
        pair_matrix,
        overlap_constraints=previous_constraints,
    )
    set_2, fitness_2 = exact_best_set(
        final_probabilities,
        pair_matrix,
        forbidden_set=set_1,
        max_overlap=CFG.max_overlap,
        overlap_constraints=previous_constraints,
    )
    ranking_indexes = np.argsort(final_probabilities)[::-1]
    ranking = [(int(i + 1), float(final_probabilities[i])) for i in ranking_indexes]
    return Prediction(
        next_draw_number, pd.Timestamp(next_draw_date), set_1, set_2,
        fitness_1, fitness_2, ranking
    )


# ============================================================
# 9. BUNDLE, TRENING GŁÓWNY I PREDYKCJA
# ============================================================

@dataclass
class PredictorBundle:
    model: nn.Module
    scaler: StandardScaler
    calibrator: LogisticRegression
    ensemble_weight: float
    validation_metrics: EvaluationMetrics
    test_metrics: EvaluationMetrics
    random_metrics: EvaluationMetrics
    modern_history: pd.DataFrame
    history_matrix: np.ndarray
    input_dim: int
    hybrid_model: Optional[nn.Module] = None
    hybrid_calibrator: Optional[LogisticRegression] = None
    hybrid_weight: float = 0.0
    base_validation_metrics: Optional[EvaluationMetrics] = None
    hybrid_validation_metrics: Optional[EvaluationMetrics] = None


def train_predictor(history_df: pd.DataFrame) -> PredictorBundle:
    print("\n==============================")
    print("MINI LOTTO MODEL TRAINING")
    print("==============================")
    modern = get_modern_history(history_df)
    history = history_to_matrix(modern)
    X, Y = build_dataset(modern)
    data = temporal_split(X, Y)
    print(f"Współczesne losowania: {len(modern)}")
    print(f"Dataset X: {X.shape}, Y: {Y.shape}, device: {CFG.device}")

    set_seed(SEED)
    training = train_model(
        data.X_train, data.Y_train, data.X_val, data.Y_val, architecture="mlp"
    )
    print(
        f"MLP — epoki: {training.epochs_run}, "
        f"best val loss: {training.best_val_loss:.6f}"
    )

    set_seed(SEED + 1000)
    hybrid_training = train_model(
        data.X_train, data.Y_train, data.X_val, data.Y_val, architecture="hybrid"
    )
    print(
        f"Temporalny CNN+MLP — epoki: {hybrid_training.epochs_run}, "
        f"best val loss: {hybrid_training.best_val_loss:.6f}"
    )

    val_logits = model_logits(training.model, data.X_val)
    test_logits = model_logits(training.model, data.X_test)
    calibrator = fit_platt_calibrator(val_logits, data.Y_val)
    nn_val = calibrated_nn_probabilities(calibrator, val_logits)
    nn_test = calibrated_nn_probabilities(calibrator, test_logits)

    hybrid_val_logits = model_logits(hybrid_training.model, data.X_val)
    hybrid_test_logits = model_logits(hybrid_training.model, data.X_test)
    hybrid_calibrator = fit_platt_calibrator(hybrid_val_logits, data.Y_val)
    hybrid_val = calibrated_nn_probabilities(hybrid_calibrator, hybrid_val_logits)
    hybrid_test = calibrated_nn_probabilities(hybrid_calibrator, hybrid_test_logits)

    offset = dataset_offset()
    val_history_start = offset + data.val_start_dataset
    test_history_start = offset + data.test_start_dataset
    stat_val = statistical_predictions(history, val_history_start, test_history_start)
    stat_test = statistical_predictions(history, test_history_start, len(history))

    weight = optimize_ensemble_weight(nn_val, stat_val, data.Y_val)
    base_val_predictions = ensemble_probabilities(nn_val, stat_val, weight)
    base_test_predictions = ensemble_probabilities(nn_test, stat_test, weight)
    hybrid_weight = optimize_hybrid_weight(
        base_val_predictions, hybrid_val, data.Y_val
    )
    val_predictions = blend_hybrid_probabilities(
        base_val_predictions, hybrid_val, hybrid_weight
    )
    test_predictions = blend_hybrid_probabilities(
        base_test_predictions, hybrid_test, hybrid_weight
    )
    base_validation_metrics = evaluate_scores(base_val_predictions, data.Y_val)
    hybrid_validation_metrics = evaluate_scores(hybrid_val, data.Y_val)
    validation_metrics = evaluate_scores(val_predictions, data.Y_val)
    test_metrics = evaluate_scores(test_predictions, data.Y_test)
    random_metrics = analytical_random_baseline()

    print(f"Wybrana waga MLP względem statystyki: {weight:.2f}")
    print(f"Dopuszczona waga temporalnego CNN: {hybrid_weight:.2f}")
    print_metrics("VALIDATION — BAZOWE MLP+STAT", base_validation_metrics)
    print_metrics("VALIDATION — SAM TEMPORALNY CNN+MLP", hybrid_validation_metrics)
    print_metrics("VALIDATION — KOŃCOWY ENSEMBLE", validation_metrics)
    print_metrics("TEST — KOŃCOWY ENSEMBLE", test_metrics)
    print_metrics("ANALITYCZNY RANDOM BASELINE", random_metrics)

    return PredictorBundle(
        training.model, data.scaler, calibrator, weight,
        validation_metrics, test_metrics, random_metrics,
        modern, history, X.shape[1],
        hybrid_training.model, hybrid_calibrator, hybrid_weight,
        base_validation_metrics, hybrid_validation_metrics,
    )


def infer_next_draw_date(modern_history: pd.DataFrame) -> pd.Timestamp:
    # W aktualnym Mini Lotto losowania odbywają się codziennie. Parametr można
    # jawnie nadpisać w main(), co jest bezpieczniejsze dla archiwalnego snapshotu.
    return pd.Timestamp(modern_history.iloc[-1]["Date"]) + pd.Timedelta(days=1)


def build_next_draw_features(df: pd.DataFrame, next_draw_date: pd.Timestamp) -> np.ndarray:
    history = history_to_matrix(df)
    return build_feature_vector(df, history, len(history), pd.Timestamp(next_draw_date))


def predict_next_draw(
    bundle: PredictorBundle,
    next_draw_date: Optional[pd.Timestamp] = None,
    previous_prediction: Optional["StoredPrediction"] = None,
) -> Prediction:
    if next_draw_date is None:
        next_draw_date = infer_next_draw_date(bundle.modern_history)
    features = build_next_draw_features(bundle.modern_history, pd.Timestamp(next_draw_date))
    scaled = bundle.scaler.transform(features.reshape(1, -1)).astype(np.float32)
    logits = model_logits(bundle.model, scaled)
    nn_probability = calibrated_nn_probabilities(bundle.calibrator, logits)[0]
    stat_probability = statistical_scores(bundle.history_matrix, len(bundle.history_matrix))
    final_probability = ensemble_probabilities(
        nn_probability, stat_probability, bundle.ensemble_weight
    )
    if (
        bundle.hybrid_model is not None
        and bundle.hybrid_calibrator is not None
        and bundle.hybrid_weight > 0.0
    ):
        hybrid_logits = model_logits(bundle.hybrid_model, scaled)
        hybrid_probability = calibrated_nn_probabilities(
            bundle.hybrid_calibrator, hybrid_logits
        )[0]
        final_probability = blend_hybrid_probabilities(
            final_probability, hybrid_probability, bundle.hybrid_weight
        )
    last_number = int(bundle.modern_history.iloc[-1]["Numer"])
    prediction = generate_two_sets(
        final_probability,
        bundle.history_matrix,
        last_number + 1,
        pd.Timestamp(next_draw_date),
        previous_prediction=previous_prediction,
    )
    prediction.hybrid_weight = float(bundle.hybrid_weight)
    return prediction


def print_prediction(prediction: Prediction, top_n: int = 15) -> None:
    print("\n==============================")
    print(f"PREDYKCJA LOSOWANIA {prediction.next_draw_number}")
    print(f"DATA: {prediction.next_draw_date.date()}")
    print("==============================")
    print(f"ZESTAW 1: {prediction.set_1}")
    print(f"ZESTAW 2: {prediction.set_2}")
    overlap = len(set(prediction.set_1) & set(prediction.set_2))
    print(f"Overlap: {overlap}")
    print("\nTOP RANKING (score modelowy, nie gwarancja trafienia):")
    for rank, (number, score) in enumerate(prediction.ranking[:top_n], 1):
        print(f"{rank:2d}. {number:2d}  {score:.6f}")


# ============================================================
# 10. EXPANDING-WINDOW WALK-FORWARD Z PONOWNYM TRENINGIEM
# ============================================================

@dataclass
class WalkForwardResult:
    metrics: EvaluationMetrics
    fold_metrics: List[EvaluationMetrics]
    observations: int


def expanding_walk_forward(
    modern: pd.DataFrame, X: np.ndarray, Y: np.ndarray
) -> WalkForwardResult:
    """
    Każdy fold:
      * trenuje nową sieć tylko na danych sprzed folda,
      * kalibruje i dobiera wagę na końcówce dostępnej historii,
      * ocenia kolejny, niewidziany blok losowań.
    """
    history = history_to_matrix(modern)
    total_test = CFG.walk_forward_folds * CFG.walk_forward_block_size
    first_test_start = len(X) - total_test
    if first_test_start <= CFG.walk_forward_validation_size + CFG.batch_size:
        raise ValueError("Za mało danych do skonfigurowanego walk-forward.")

    all_predictions, all_targets, folds = [], [], []
    offset = dataset_offset()
    print("\n==============================")
    print("EXPANDING-WINDOW WALK-FORWARD")
    print("==============================")

    for fold in range(CFG.walk_forward_folds):
        evaluation_start = first_test_start + fold * CFG.walk_forward_block_size
        evaluation_end = evaluation_start + CFG.walk_forward_block_size
        validation_start = evaluation_start - CFG.walk_forward_validation_size

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[:validation_start]).astype(np.float32)
        X_val = scaler.transform(X[validation_start:evaluation_start]).astype(np.float32)
        X_eval = scaler.transform(X[evaluation_start:evaluation_end]).astype(np.float32)
        Y_train = Y[:validation_start]
        Y_val = Y[validation_start:evaluation_start]
        Y_eval = Y[evaluation_start:evaluation_end]

        set_seed(SEED + fold + 1)
        trained = train_model(
            X_train, Y_train, X_val, Y_val,
            epochs=CFG.walk_forward_epochs,
            patience=CFG.walk_forward_patience,
            architecture="mlp",
        )
        val_logits = model_logits(trained.model, X_val)
        eval_logits = model_logits(trained.model, X_eval)
        calibrator = fit_platt_calibrator(val_logits, Y_val)
        nn_val = calibrated_nn_probabilities(calibrator, val_logits)
        nn_eval = calibrated_nn_probabilities(calibrator, eval_logits)

        set_seed(SEED + 1000 + fold + 1)
        hybrid_trained = train_model(
            X_train, Y_train, X_val, Y_val,
            epochs=CFG.walk_forward_epochs,
            patience=CFG.walk_forward_patience,
            architecture="hybrid",
        )
        hybrid_val_logits = model_logits(hybrid_trained.model, X_val)
        hybrid_eval_logits = model_logits(hybrid_trained.model, X_eval)
        hybrid_calibrator = fit_platt_calibrator(hybrid_val_logits, Y_val)
        hybrid_val = calibrated_nn_probabilities(
            hybrid_calibrator, hybrid_val_logits
        )
        hybrid_eval = calibrated_nn_probabilities(
            hybrid_calibrator, hybrid_eval_logits
        )

        stat_val = statistical_predictions(
            history, offset + validation_start, offset + evaluation_start
        )
        stat_eval = statistical_predictions(
            history, offset + evaluation_start, offset + evaluation_end
        )
        weight = optimize_ensemble_weight(nn_val, stat_val, Y_val)
        base_val = ensemble_probabilities(nn_val, stat_val, weight)
        base_eval = ensemble_probabilities(nn_eval, stat_eval, weight)
        hybrid_weight = optimize_hybrid_weight(base_val, hybrid_val, Y_val)
        predictions = blend_hybrid_probabilities(
            base_eval, hybrid_eval, hybrid_weight
        )
        metrics = evaluate_scores(predictions, Y_eval)
        folds.append(metrics)
        all_predictions.append(predictions)
        all_targets.append(Y_eval)
        print(
            f"Fold {fold + 1}/{CFG.walk_forward_folds}: "
            f"zakres dataset [{evaluation_start}:{evaluation_end}), "
            f"waga MLP={weight:.2f}, waga CNN={hybrid_weight:.2f}, "
            f"mean={metrics.mean_hits:.3f}, "
            f">=3/5={100 * metrics.rate_ge_3:.3f}%"
        )

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    total_metrics = evaluate_scores(predictions, targets)
    print_metrics("WALK-FORWARD ŁĄCZNIE", total_metrics)
    return WalkForwardResult(total_metrics, folds, len(targets))


# ============================================================
# 11. AUDYT POPRZEDNIEJ PREDYKCJI
# ============================================================

@dataclass
class PredictionAudit:
    actual_numbers: set[int]
    set1_hits: int
    set2_hits: int
    best_hits: int
    set1_false_positive: set[int]
    set1_false_negative: set[int]
    set2_false_positive: set[int]
    set2_false_negative: set[int]


def audit_prediction(prediction: Prediction, actual_numbers: Sequence[int]) -> PredictionAudit:
    if len(actual_numbers) != CFG.draw_size or len(set(actual_numbers)) != CFG.draw_size:
        raise ValueError("Rzeczywisty wynik musi zawierać 5 różnych liczb.")
    actual = set(map(int, actual_numbers))
    set_1, set_2 = set(prediction.set_1), set(prediction.set_2)
    hit_1, hit_2 = len(set_1 & actual), len(set_2 & actual)
    return PredictionAudit(
        actual, hit_1, hit_2, max(hit_1, hit_2),
        set_1 - actual, actual - set_1, set_2 - actual, actual - set_2
    )


@dataclass
class StoredPrediction:
    """Minimalny, trwały zapis typów potrzebny do audytu po losowaniu."""

    draw_number: int
    draw_date: pd.Timestamp
    set_1: Individual
    set_2: Individual


def prediction_state_path(bundle_path: str) -> Path:
    """Umieszcza stan predykcji obok bundle, także na Dysku Google."""

    bundle = Path(bundle_path)
    return bundle.with_name(f"{bundle.stem}_last_prediction.json")


def save_prediction_state(prediction: Prediction, path: Path) -> None:
    """Zapisuje typy atomowo, aby przerwane zapisy nie uszkodziły stanu."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "draw_number": int(prediction.next_draw_number),
        "draw_date": prediction.next_draw_date.date().isoformat(),
        "set_1": [int(number) for number in prediction.set_1],
        "set_2": [int(number) for number in prediction.set_2],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    print(f"Predykcja do późniejszego audytu zapisana: {path.resolve()}")


def load_prediction_state(path: Path) -> Optional[StoredPrediction]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("nieobsługiwana wersja pliku stanu")
        set_1 = tuple(sorted(int(number) for number in payload["set_1"]))
        set_2 = tuple(sorted(int(number) for number in payload["set_2"]))
        for numbers in (set_1, set_2):
            if len(numbers) != CFG.draw_size or len(set(numbers)) != CFG.draw_size:
                raise ValueError("zapisany zestaw nie zawiera 5 różnych liczb")
            if min(numbers) < 1 or max(numbers) > CFG.n_numbers:
                raise ValueError("zapisany zestaw jest poza zakresem 1..42")
        return StoredPrediction(
            draw_number=int(payload["draw_number"]),
            draw_date=pd.Timestamp(payload["draw_date"]),
            set_1=set_1,
            set_2=set_2,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        warnings.warn(f"Pomijam niepoprawny plik stanu predykcji {path}: {exc}")
        return None


def _draw_numbers(row: pd.Series) -> List[int]:
    return sorted(int(row[column]) for column in NUMBER_COLUMNS)


def analyze_latest_draw(history: pd.DataFrame) -> dict:
    """Opisuje ostatnie losowanie na tle wyłącznie wcześniejszej historii."""

    modern = get_modern_history(history)
    latest = modern.iloc[-1]
    numbers = _draw_numbers(latest)
    before = modern.iloc[:-1].reset_index(drop=True)
    before_matrix = history_to_matrix(before) if len(before) else np.empty((0, CFG.n_numbers))

    previous_numbers: set[int] = set()
    if len(modern) >= 2:
        previous_numbers = set(_draw_numbers(modern.iloc[-2]))

    gaps_before = {}
    frequencies_50 = {}
    recent = before_matrix[-50:]
    for number in numbers:
        index = number - 1
        appearances = np.flatnonzero(before_matrix[:, index] > 0)
        gaps_before[number] = (
            len(before_matrix) + 1
            if len(appearances) == 0
            else len(before_matrix) - int(appearances[-1])
        )
        frequencies_50[number] = (
            0.0 if len(recent) == 0 else float(recent[:, index].mean())
        )

    consecutive_pairs = [
        (left, right)
        for left, right in zip(numbers, numbers[1:])
        if right - left == 1
    ]
    return {
        "draw_number": int(latest["Numer"]),
        "draw_date": pd.Timestamp(latest["Date"]),
        "numbers": numbers,
        "sum": int(sum(numbers)),
        "odd": int(sum(number % 2 == 1 for number in numbers)),
        "even": int(sum(number % 2 == 0 for number in numbers)),
        "low": int(sum(number <= CFG.n_numbers // 2 for number in numbers)),
        "high": int(sum(number > CFG.n_numbers // 2 for number in numbers)),
        "spread": int(max(numbers) - min(numbers)),
        "consecutive_pairs": consecutive_pairs,
        "repeated_from_previous": sorted(set(numbers) & previous_numbers),
        "gaps_before": gaps_before,
        "frequencies_50": frequencies_50,
    }


def audit_stored_prediction(
    history: pd.DataFrame,
    stored: Optional[StoredPrediction],
) -> Optional[PredictionAudit]:
    if stored is None:
        return None
    matching = history.loc[history["Numer"] == stored.draw_number]
    if matching.empty:
        return None
    actual_numbers = _draw_numbers(matching.iloc[-1])
    return audit_prediction(stored, actual_numbers)


PREDICTION_HISTORY_COLUMNS = [
    "draw_number",
    "draw_date",
    "set_1",
    "set_2",
    "source_draw_number",
    "created_at_utc",
    "feature_version",
    "hybrid_weight",
    "actual_numbers",
    "set_1_hits",
    "set_2_hits",
    "audited_at_utc",
]


def _csv_numbers(numbers: Sequence[int]) -> str:
    return " ".join(str(int(number)) for number in sorted(numbers))


def update_prediction_history(
    path: Path,
    history: pd.DataFrame,
    previous_prediction: Optional[StoredPrediction],
    next_prediction: Prediction,
) -> None:
    """Aktualizuje audyt poprzedniej prognozy i dopisuje następną."""

    rows: List[dict] = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    now = datetime.now(timezone.utc).isoformat()
    audit = audit_stored_prediction(history, previous_prediction)
    if previous_prediction is not None and audit is not None:
        previous_key = str(previous_prediction.draw_number)
        matching = next(
            (row for row in rows if row.get("draw_number") == previous_key), None
        )
        if matching is None:
            matching = {
                column: "" for column in PREDICTION_HISTORY_COLUMNS
            }
            matching.update(
                {
                    "draw_number": previous_key,
                    "draw_date": previous_prediction.draw_date.date().isoformat(),
                    "set_1": _csv_numbers(previous_prediction.set_1),
                    "set_2": _csv_numbers(previous_prediction.set_2),
                }
            )
            rows.append(matching)
        matching.update(
            {
                "actual_numbers": _csv_numbers(audit.actual_numbers),
                "set_1_hits": str(audit.set1_hits),
                "set_2_hits": str(audit.set2_hits),
                "audited_at_utc": now,
            }
        )

    next_key = str(next_prediction.next_draw_number)
    rows = [row for row in rows if row.get("draw_number") != next_key]
    source_draw_number = int(get_modern_history(history).iloc[-1]["Numer"])
    rows.append(
        {
            "draw_number": next_key,
            "draw_date": next_prediction.next_draw_date.date().isoformat(),
            "set_1": _csv_numbers(next_prediction.set_1),
            "set_2": _csv_numbers(next_prediction.set_2),
            "source_draw_number": str(source_draw_number),
            "created_at_utc": now,
            "feature_version": str(FEATURE_VERSION),
            "hybrid_weight": f"{next_prediction.hybrid_weight:.4f}",
            "actual_numbers": "",
            "set_1_hits": "",
            "set_2_hits": "",
            "audited_at_utc": "",
        }
    )
    rows.sort(key=lambda row: int(row["draw_number"]))

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    print(f"Historia prognoz i audytów zapisana: {path.resolve()}")


def _format_numbers(numbers: Sequence[int]) -> str:
    return " ".join(f"{int(number):02d}" for number in sorted(numbers))


def _emit_complete_draw_summary(
    history: pd.DataFrame,
    previous_prediction: Optional[StoredPrediction],
    next_prediction: Prediction,
    top_n: int = 15,
) -> None:
    """Jeden raport: analiza wyniku, audyt typów i dwa kolejne zestawy."""

    analysis = analyze_latest_draw(history)
    audit = audit_stored_prediction(history, previous_prediction)

    print("\n" + "=" * 62)
    print(f"PODSUMOWANIE PO LOSOWANIU MINI LOTTO {analysis['draw_number']}")
    print(f"DATA LOSOWANIA: {analysis['draw_date'].date()}")
    print("=" * 62)
    print(f"WYLOSOWANE LICZBY: {_format_numbers(analysis['numbers'])}")
    print(f"Suma: {analysis['sum']} | Rozstęp: {analysis['spread']}")
    print(
        f"Nieparzyste/parzyste: {analysis['odd']}/{analysis['even']} | "
        f"Zakres 1–21/22–42: {analysis['low']}/{analysis['high']}"
    )
    pairs = analysis["consecutive_pairs"]
    print(
        "Pary kolejne: "
        + (", ".join(f"{a:02d}-{b:02d}" for a, b in pairs) if pairs else "brak")
    )
    repeats = analysis["repeated_from_previous"]
    print(
        "Powtórzone z poprzedniego losowania: "
        + (_format_numbers(repeats) if repeats else "brak")
    )
    print(
        "Przerwy przed wystąpieniem: "
        + ", ".join(
            f"{number:02d}: {analysis['gaps_before'][number]}"
            for number in analysis["numbers"]
        )
    )
    print(
        "Częstość w poprzednich 50 losowaniach: "
        + ", ".join(
            f"{number:02d}: {100 * analysis['frequencies_50'][number]:.1f}%"
            for number in analysis["numbers"]
        )
    )

    print("\nOCENA POPRZEDNIEJ PROGNOZY")
    print("-" * 62)
    if previous_prediction is None:
        print("Brak wcześniejszego pliku predykcji — audyt rozpocznie się od następnego losowania.")
    elif audit is None:
        print(
            f"Zapisane zestawy oczekują na losowanie "
            f"{previous_prediction.draw_number}."
        )
    else:
        print(f"Oceniane losowanie: {previous_prediction.draw_number}")
        hit_numbers_1 = set(previous_prediction.set_1) & audit.actual_numbers
        hit_numbers_2 = set(previous_prediction.set_2) & audit.actual_numbers
        print(
            f"ZESTAW 1: {_format_numbers(previous_prediction.set_1)} | "
            f"trafienia: {audit.set1_hits} "
            f"({_format_numbers(hit_numbers_1) if hit_numbers_1 else 'brak'})"
        )
        print(
            f"ZESTAW 2: {_format_numbers(previous_prediction.set_2)} | "
            f"trafienia: {audit.set2_hits} "
            f"({_format_numbers(hit_numbers_2) if hit_numbers_2 else 'brak'})"
        )
        print(f"Najlepszy wynik dwóch zestawów: {audit.best_hits}/5")

    print("\nWNIOSKI I REAKCJA MODELU")
    print("-" * 62)
    print("Do nowej prognozy użyto historii obejmującej ocenione losowanie.")
    if audit is None:
        print("Brak zakończonego audytu, więc nie zastosowano oceny trafień.")
    else:
        print(
            f"Audyt: zestaw 1 trafił {audit.set1_hits}/5, "
            f"zestaw 2 trafił {audit.set2_hits}/5."
        )
        print(
            "Filtr rotacji nie pozwala powtórzyć poprzednich zestawów "
            "i wymusza zmianę co najmniej 2 liczb w każdym nowym zestawie."
        )

    print("\n" + "=" * 62)
    print(f"PROGNOZA NASTĘPNEGO LOSOWANIA MINI LOTTO {next_prediction.next_draw_number}")
    print(f"PRZEWIDYWANA DATA: {next_prediction.next_draw_date.date()}")
    print("=" * 62)
    print(f"ZESTAW 1: {_format_numbers(next_prediction.set_1)}")
    print(f"ZESTAW 2: {_format_numbers(next_prediction.set_2)}")
    if next_prediction.hybrid_weight > 0.0:
        print(
            "Temporalny CNN+MLP: aktywny w ensemble, udział "
            f"{100 * next_prediction.hybrid_weight:.0f}%"
        )
    else:
        print(
            "Temporalny CNN+MLP: udział 0% — nie wykazał wymaganej "
            "poprawy Brier na walidacji"
        )
    overlap = len(set(next_prediction.set_1) & set(next_prediction.set_2))
    print(f"Wspólne liczby zestawów: {overlap}")
    if previous_prediction is not None and audit is not None:
        previous_sets = (set(previous_prediction.set_1), set(previous_prediction.set_2))
        for index, new_set in enumerate((next_prediction.set_1, next_prediction.set_2), 1):
            max_previous_common = max(
                len(set(new_set) & old_set) for old_set in previous_sets
            )
            print(
                f"Zmiana zestawu {index}: co najmniej "
                f"{CFG.draw_size - max_previous_common} liczb względem "
                "każdego poprzedniego zestawu"
            )
    print("\nTOP RANKING (score modelowy, nie gwarancja trafienia):")
    for rank, (number, score) in enumerate(next_prediction.ranking[:top_n], 1):
        print(f"{rank:2d}. {number:02d}  {score:.6f}")


def print_complete_draw_summary(
    history: pd.DataFrame,
    previous_prediction: Optional[StoredPrediction],
    next_prediction: Prediction,
    top_n: int = 15,
) -> str:
    """Wyświetla raport i zwraca jego tekst do zapisu przez automatyzację."""

    buffer = StringIO()
    with redirect_stdout(buffer):
        _emit_complete_draw_summary(
            history, previous_prediction, next_prediction, top_n=top_n
        )
    report = buffer.getvalue()
    print(report, end="")
    return report


# ============================================================
# 12. PEŁNY ZAPIS I ODCZYT BUNDLE
# ============================================================

def save_bundle(bundle: PredictorBundle, path: str = "mini_lotto_bundle.pt") -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_version": FEATURE_VERSION,
        "config": asdict(CFG),
        "input_dim": bundle.input_dim,
        "model_state_dict": bundle.model.state_dict(),
        "scaler": bundle.scaler,
        "calibrator": bundle.calibrator,
        "ensemble_weight": bundle.ensemble_weight,
        "validation_metrics": bundle.validation_metrics,
        "test_metrics": bundle.test_metrics,
        "random_metrics": bundle.random_metrics,
        "modern_history": bundle.modern_history,
        "history_matrix": bundle.history_matrix,
        "hybrid_model_state_dict": (
            bundle.hybrid_model.state_dict() if bundle.hybrid_model is not None else None
        ),
        "hybrid_calibrator": bundle.hybrid_calibrator,
        "hybrid_weight": float(bundle.hybrid_weight),
        "base_validation_metrics": bundle.base_validation_metrics,
        "hybrid_validation_metrics": bundle.hybrid_validation_metrics,
    }
    torch.save(payload, destination)
    print(f"Pełny bundle zapisany: {destination.resolve()}")


def load_bundle(path: str = "mini_lotto_bundle.pt") -> PredictorBundle:
    # weights_only=False jest konieczne, bo bundle zawiera scaler, kalibrator i DataFrame.
    # Wczytuj wyłącznie własne, zaufane pliki .pt.
    checkpoint = torch.load(path, map_location=CFG.device, weights_only=False)
    if checkpoint.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"Niezgodna wersja cech: {checkpoint.get('feature_version')} "
            f"zamiast {FEATURE_VERSION}."
        )
    saved_config = checkpoint.get("config", {})
    # Odtwarzamy parametry wpływające na architekturę i cechy. Urządzenia nie
    # odtwarzamy: model ma działać także po przeniesieniu z GPU na CPU.
    for field_name in Config.__dataclass_fields__:
        if field_name != "device" and field_name in saved_config:
            value = saved_config[field_name]
            if field_name == "rolling_windows":
                value = tuple(value)
            setattr(CFG, field_name, value)
    model = MiniLottoMLP(int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(CFG.device).eval()
    hybrid_model: Optional[nn.Module] = None
    hybrid_state = checkpoint.get("hybrid_model_state_dict")
    if hybrid_state is not None:
        hybrid_model = MiniLottoTemporalHybrid(int(checkpoint["input_dim"]))
        hybrid_model.load_state_dict(hybrid_state)
        hybrid_model.to(CFG.device).eval()
    return PredictorBundle(
        model=model,
        scaler=checkpoint["scaler"],
        calibrator=checkpoint["calibrator"],
        ensemble_weight=float(checkpoint["ensemble_weight"]),
        validation_metrics=checkpoint["validation_metrics"],
        test_metrics=checkpoint["test_metrics"],
        random_metrics=checkpoint["random_metrics"],
        modern_history=checkpoint["modern_history"],
        history_matrix=checkpoint["history_matrix"],
        input_dim=int(checkpoint["input_dim"]),
        hybrid_model=hybrid_model,
        hybrid_calibrator=checkpoint.get("hybrid_calibrator"),
        hybrid_weight=float(checkpoint.get("hybrid_weight", 0.0)),
        base_validation_metrics=checkpoint.get("base_validation_metrics"),
        hybrid_validation_metrics=checkpoint.get("hybrid_validation_metrics"),
    )


def refresh_bundle_history(
    bundle: PredictorBundle,
    history: pd.DataFrame,
) -> PredictorBundle:
    """Podmienia historię bez ponownego trenowania modelu."""

    modern = get_modern_history(history)
    next_features = build_next_draw_features(modern, infer_next_draw_date(modern))
    if len(next_features) != bundle.input_dim:
        raise ValueError(
            f"Model oczekuje {bundle.input_dim} cech, a aktualny kod tworzy "
            f"{len(next_features)}. Wykonaj pełny trening."
        )
    bundle.modern_history = modern
    bundle.history_matrix = history_to_matrix(modern)
    return bundle


# ============================================================
# 13. MAIN
# ============================================================

def resolve_csv_path(explicit_path: Optional[str] = None) -> str:
    candidates = [
        explicit_path,
        "/content/drive/MyDrive/MiniLotto/wyniki-minilotto.csv",
        "/content/wyniki-minilotto.csv",       # Google Colab
        "/mnt/data/wyniki-minilotto.csv",      # środowiska notebookowe
        "wyniki-minilotto.csv",                # bieżący katalog
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return str(candidate)
    checked = [candidate for candidate in candidates if candidate]
    raise FileNotFoundError("Nie znaleziono CSV. Sprawdzone ścieżki: " + ", ".join(checked))


def main(
    csv_path: Optional[str] = None,
    next_draw_date: Optional[str] = None,
    bundle_path: str = "mini_lotto_bundle.pt",
    mode: str = "train",
    state_path: Optional[str] = None,
    report_path: Optional[str] = None,
    prediction_history_path: Optional[str] = None,
) -> Prediction:
    mode = mode.lower().strip()
    if mode not in {"daily", "train"}:
        raise ValueError("Tryb musi mieć wartość 'daily' albo 'train'.")
    warnings.filterwarnings("default")
    set_seed(SEED)
    selected_path = resolve_csv_path(csv_path)
    history = load_history(selected_path)
    state_file = Path(state_path) if state_path else prediction_state_path(bundle_path)
    prediction_history_file = (
        Path(prediction_history_path)
        if prediction_history_path
        else state_file.with_name("prediction_history.csv")
    )
    previous_prediction = load_prediction_state(state_file)
    print(f"Plik: {selected_path}")
    print(f"Tryb: {'codzienna predykcja' if mode == 'daily' else 'pełny trening'}")
    print(f"Pełny snapshot: {len(history)} losowań")
    print("Ostatni rekord:")
    print(history[["Numer", "Date", *NUMBER_COLUMNS]].tail(1).to_string(index=False))

    latest_draw_number = int(get_modern_history(history).iloc[-1]["Numer"])
    prediction_is_pending = (
        previous_prediction is not None
        and previous_prediction.draw_number > latest_draw_number
    )
    if prediction_is_pending and mode == "daily":
        print(
            f"Brak nowego losowania: prognoza na losowanie "
            f"{previous_prediction.draw_number} nadal oczekuje na wynik. "
            "Nie zmieniam zestawów ani pliku stanu."
        )
        return Prediction(
            previous_prediction.draw_number,
            previous_prediction.draw_date,
            previous_prediction.set_1,
            previous_prediction.set_2,
            float("nan"),
            float("nan"),
            [],
        )

    if mode == "daily":
        if not Path(bundle_path).is_file():
            raise FileNotFoundError(
                f"Brak zapisanego modelu {bundle_path}. Najpierw uruchom tryb train."
            )
        bundle = refresh_bundle_history(load_bundle(bundle_path), history)
        print("Wczytano zapisany model; pominięto trening i walk-forward.")
    else:
        modern = get_modern_history(history)
        X, Y = build_dataset(modern)
        if CFG.walk_forward_enabled:
            expanding_walk_forward(modern, X, Y)

        # Ponownie ustawiamy bazowe ziarno, aby wynik głównego modelu nie zależał
        # od liczby foldów walk-forward.
        set_seed(SEED)
        bundle = train_predictor(history)
        save_bundle(bundle, bundle_path)

    # Pełny trening powinien dojść do skutku także wtedy, gdy czekamy jeszcze
    # na wynik prognozowanego losowania. Po zapisaniu nowego bundle kończymy
    # jednak bez tworzenia kolejnych zestawów i bez nadpisywania raportu/stanu.
    if prediction_is_pending:
        assert previous_prediction is not None
        print(
            f"Model został wytrenowany i zapisany, ale prognoza na losowanie "
            f"{previous_prediction.draw_number} nadal oczekuje na wynik. "
            "Nie zmieniam zestawów, raportu ani pliku stanu."
        )
        return Prediction(
            previous_prediction.draw_number,
            previous_prediction.draw_date,
            previous_prediction.set_1,
            previous_prediction.set_2,
            float("nan"),
            float("nan"),
            [],
        )

    explicit_date = pd.Timestamp(next_draw_date) if next_draw_date else None
    prediction = predict_next_draw(
        bundle,
        explicit_date,
        previous_prediction=previous_prediction,
    )
    report = print_complete_draw_summary(history, previous_prediction, prediction)
    update_prediction_history(
        prediction_history_file, history, previous_prediction, prediction
    )
    save_prediction_state(prediction, state_file)
    if report_path:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(report.lstrip(), encoding="utf-8")
        os.replace(temporary, destination)
        print(f"Raport zapisany: {destination.resolve()}")
    return prediction


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analiza i predykcja Mini Lotto")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--next-draw-date", default=None)
    parser.add_argument("--bundle-path", default=None)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--prediction-history-path", default=None)
    parser.add_argument(
        "--mode", choices=("daily", "train"), default="train",
        help="daily używa zapisanego modelu; train wykonuje pełny trening",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    # Po zamontowaniu Dysku Google w Colabie model i poprzednia prognoza
    # przetrwają zamknięcie sesji. Poza Colabem używany jest katalog lokalny.
    GOOGLE_DRIVE_ROOT = Path("/content/drive/MyDrive")
    default_bundle_path = str(
        GOOGLE_DRIVE_ROOT / "MiniLotto" / "mini_lotto_bundle.pt"
        if GOOGLE_DRIVE_ROOT.is_dir()
        else Path("mini_lotto_bundle.pt")
    )
    main(
        csv_path=arguments.csv_path,
        next_draw_date=arguments.next_draw_date,
        bundle_path=arguments.bundle_path or default_bundle_path,
        mode=arguments.mode,
        state_path=arguments.state_path,
        report_path=arguments.report_path,
        prediction_history_path=arguments.prediction_history_path,
    )
