"""Causal, per-draw ranking and deterministic two-ticket selection.

PairLogit penalizes putting a non-drawn ball above a drawn ball. Rank scores
are not probabilities: a separate temporal calibration block fits their scale.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import os
import tempfile
import zipfile

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostRanker, Pool

VERSION = 1
WARMUP = 250
WINDOWS = (5, 10, 25, 50, 100, 250)
SEED = 42


@dataclass(frozen=True)
class Settings:
    iterations: int = 160
    depth: int = 4
    learning_rate: float = .05
    l2_leaf_reg: float = 8.
    loss_function: str = 'PairLogit'
    selector: str = 'greedy'


def features(matrix):
    """Return n+1 x 42 x features; row i uses only matrix[:i]."""
    y = np.asarray(matrix, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != 42 or not np.isin(y, [0, 1]).all() or not (y.sum(1) == 5).all():
        raise ValueError('Expected binary draws of five distinct balls from 42')
    n = len(y)
    cumulative = np.vstack([np.zeros((1, 42)), y.cumsum(0)])
    rows = np.arange(n+1)
    cols = [np.broadcast_to(np.arange(42)/41, (n+1, 42))]
    for window in WINDOWS:
        start = np.maximum(0, rows-window)
        cols.append((cumulative-cumulative[start]) / np.maximum(1, rows-start)[:, None])
    gaps = np.zeros((n+1, 42))
    for i in range(n):
        gaps[i+1] = np.where(y[i] == 1, 0, gaps[i]+1)
    cols.extend([np.log1p(gaps), cols[1]-cols[4],
                 np.vstack([np.zeros((1, 42)), y])])
    return np.stack(cols, axis=-1).astype(np.float32)


def pool(x, y):
    return Pool(x.reshape(-1, x.shape[-1]), y.reshape(-1),
                group_id=np.repeat(np.arange(len(y)), 42))


def fit(x, y, settings):
    model = CatBoostRanker(**{k:v for k,v in asdict(settings).items() if k != 'selector'},
                          random_seed=SEED, thread_count=2, verbose=False,
                          allow_writing_files=False)
    model.fit(pool(x, y))
    return model


def scores(model, x):
    return np.asarray(model.predict(x.reshape(-1, x.shape[-1]))).reshape(-1, 42)


def calibrate(raw, y):
    fit_cal = LogisticRegression(C=1., solver='lbfgs').fit(raw.reshape(-1, 1), y.reshape(-1))
    # A negative ranking slope is evidence to fall back to uniform probabilities.
    return [max(0., float(fit_cal.coef_[0, 0])), float(fit_cal.intercept_[0])]


def probabilities(raw, calibration):
    """Calibrated marginal estimates with exactly five expected balls."""
    z = np.asarray(raw, dtype=float)*calibration[0] + calibration[1]
    if not np.isfinite(z).all():
        raise ValueError('Non-finite model scores')
    low = -60-np.max(z, axis=-1, keepdims=True)
    high = 60-np.min(z, axis=-1, keepdims=True)
    for _ in range(55):
        mid = (low+high)/2
        too_many = expit(z+mid).sum(axis=-1, keepdims=True) > 5
        high = np.where(too_many, mid, high)
        low = np.where(too_many, low, mid)
    return expit(z+(low+high)/2)


def select_two(p, previous=None, strategy='greedy', seed=SEED):
    """Two disjoint tickets, <=3 balls shared with either previous ticket.

    Greedy concentrates high scores; balanced alternates allocation between
    tickets. Both are bounded O(42), with identical rotation constraints.
    Random tie-breaking is deterministic for a draw and avoids numeric bias.
    """
    p = np.asarray(p, dtype=float)
    if p.shape != (42,) or not np.isfinite(p).all():
        raise ValueError('Expected 42 finite scores')
    if strategy not in ('greedy', 'balanced'):
        raise ValueError('Unknown selector')
    old = [set(t) for t in previous] if previous is not None else []
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(42), -p))+1
    tickets = [[], []]
    used = set()
    turns = [0]*5+[1]*5 if strategy == 'greedy' else [0, 1]*5
    for target in turns:
        for number in order:
            number = int(number)
            proposal = set(tickets[target]) | {number}
            if number not in used and all(len(proposal & t) <= 3 for t in old):
                tickets[target].append(number)
                used.add(number)
                break
        else:
            raise RuntimeError('No feasible ticket')
    return tuple(tuple(sorted(t)) for t in tickets)


def digest(matrix):
    return hashlib.sha256(np.asarray(matrix, dtype=np.uint8).tobytes()).hexdigest()


def save(model, metadata, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        native = Path(directory)/'model.cbm'
        model.save_model(str(native))
        packed = Path(directory)/'bundle.zip'
        with zipfile.ZipFile(packed, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(native, 'model.cbm')
            archive.writestr('metadata.json', json.dumps(metadata, ensure_ascii=False))
        os.replace(packed, path)


def load(path):
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read('metadata.json'))
        if metadata.get('version') != VERSION:
            raise ValueError('Unsupported CatBoost bundle version; run train')
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory)/'model.cbm'
            native.write_bytes(archive.read('model.cbm'))
            model = CatBoostRanker()
            model.load_model(str(native))
    return model, metadata
