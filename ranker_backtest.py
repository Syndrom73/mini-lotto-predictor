"""Locked chronological holdout. Selection/calibration never see test outcomes."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from math import comb, sqrt
from pathlib import Path
import time
import sys
import platform
import importlib.metadata

import numpy as np
import catboost_ranker as r


def summary(hits):
    a = np.asarray(hits, dtype=int)
    n = len(a)
    successes = int((a.max(1) >= 3).sum())
    p = successes/n
    z = 1.959963984540054
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {'draws': n, 'any_set_ge3_count': successes, 'any_set_ge3': p,
            'wilson_95': [center-radius, center+radius],
            'mean_hits': a.mean(0).tolist(),
            'hit_histograms': [np.bincount(a[:,i], minlength=6).tolist() for i in range(2)]}


def evaluate(p, y, numbers, selector, previous=None):
    hits, forecasts = [], []
    for probability, actual, draw in zip(p, y, numbers):
        tickets = r.select_two(probability, previous, selector, seed=r.SEED+int(draw))
        hits.append([int(actual[np.array(t)-1].sum()) for t in tickets])
        forecasts.append({'draw': int(draw), 'sets': tickets, 'hits': hits[-1]})
        previous = tickets
    result = summary(hits)
    result['brier'] = float(np.mean((p-y)**2))
    return result, forecasts, previous


def choose(x, y, numbers, stop, validation=200, iterations=160):
    """Train / calibration / selector selection, all strictly before stop."""
    start = stop-validation
    middle = start+validation//2
    if validation < 20 or start <= r.WARMUP+100:
        raise ValueError('Insufficient history for chronological train/calibration/validation')
    candidates = []
    best = None
    for loss in ('PairLogit', 'YetiRank:mode=NDCG;top=5'):
        settings = r.Settings(iterations=iterations, loss_function=loss)
        t = time.monotonic()
        model = r.fit(x[r.WARMUP:start], y[r.WARMUP:start], settings)
        raw = r.scores(model, x[start:stop])
        calibration = r.calibrate(raw[:middle-start], y[start:middle])
        p = r.probabilities(raw[middle-start:], calibration)
        for selector in ('greedy', 'balanced'):
            metrics, _, _ = evaluate(p, y[middle:stop], numbers[middle:stop], selector)
            candidate = {'settings': asdict(replace(settings, selector=selector)),
                         'validation': metrics, 'fit_seconds': time.monotonic()-t}
            candidates.append(candidate)
            key = (metrics['any_set_ge3'], sum(metrics['mean_hits']), -metrics['brier'])
            if best is None or key > best[0]:
                best = (key, model, calibration, replace(settings, selector=selector))
        print(f'Validation complete: {loss}', flush=True)
    return best[1:], candidates


def benchmark(history, output, test_size=400, validation=200, iterations=160, legacy=False):
    import mini_lotto_predictor as m
    modern = m.get_modern_history(history)
    y = m.history_to_matrix(modern)
    x = r.features(y)
    numbers = modern.Numer.to_numpy()
    stop = len(y)-test_size
    if test_size < 20:
        raise ValueError('Test must contain at least 20 draws')
    t = time.monotonic()
    (model, calibration, settings), candidates = choose(x, y, numbers, stop, validation, iterations)
    p = r.probabilities(r.scores(model, x[stop:-1]), calibration)
    # Start each method without previous tickets, then preserve its own state.
    metrics, forecasts, _ = evaluate(p, y[stop:], numbers[stop:], settings.selector)
    results = {'catboost': metrics}
    records = {'catboost': forecasts}
    frequency = np.array([(y[max(0,i-250):i].sum(0)+5)/(min(250,i)+42) for i in range(stop,len(y))])
    # Smoothed frequency sums to five; same ticket constraints as production.
    results['frequency'], records['frequency'], _ = evaluate(frequency, y[stop:], numbers[stop:], settings.selector)
    uniform = np.full_like(p, 5/42)
    results['random'], records['random'], _ = evaluate(uniform, y[stop:], numbers[stop:], settings.selector)
    random_rates = []
    for repeat in range(200):
        previous = None
        hits = []
        rng = np.random.default_rng(10000+repeat)
        for actual in y[stop:]:
            tickets = r.select_two(rng.random(42), previous, settings.selector)
            hits.append([int(actual[np.array(t)-1].sum()) for t in tickets])
            previous = tickets
        random_rates.append(summary(hits)['any_set_ge3'])
    if legacy:
        # Refit the legacy architecture on the same prefix, with its own earlier
        # calibration block. No loading of a present-day model into a past test.
        import model_experiments as old
        import torch
        torch.set_num_threads(2)
        old_start = time.monotonic()
        bundle = m.train_predictor(modern.iloc[:stop].copy(), finalize=False)
        old_x, _ = m.build_dataset(modern)
        old_p = old.probabilities(m, bundle, modern, old_x, stop-m.dataset_offset(), len(old_x))
        results['legacy_scores_new_selector'], records['legacy_scores_new_selector'], _ = evaluate(
            old_p, y[stop:], numbers[stop:], settings.selector)
        results['legacy_scores_new_selector']['fit_and_predict_seconds'] = time.monotonic()-old_start
    report = {'protocol_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'history_sha256': r.digest(y),
              'environment': {'python': platform.python_version(),
                  **{name: importlib.metadata.version(name) for name in ('catboost','numpy','pandas','scikit-learn','torch')}}, 'history_last_draw': int(numbers[-1]),
              'test_first_draw': int(numbers[stop]), 'test_last_draw': int(numbers[-1]),
              'test_first_date': str(modern.iloc[stop].Date.date()),
              'test_last_date': str(modern.iloc[-1].Date.date()),
              'training_last_draw': int(numbers[stop-validation-1]),
              'calibration_last_draw': int(numbers[stop-validation//2-1]),
              'selection_last_draw': int(numbers[stop-1]),
              'selected': asdict(settings), 'candidates': candidates, 'results': results,
              'random_200_runs_rate_interval': np.quantile(random_rates,[.025,.5,.975]).tolist(),
              'random_exact_any_ge3': 2*sum(comb(5,k)*comb(37,5-k) for k in range(3,6))/comb(42,5),
              'seconds': time.monotonic()-t,
              'limitations': ['Retrospective holdout, not a prospective trial; earlier project work may have examined these draws.',
                 'Test outcomes do not select parameters. Model frozen over test; features advance causally.',
                 'Legacy comparison, when enabled, uses legacy probabilities with the NEW selector; not the old full ticket policy.',
                 'Wilson interval is descriptive; overlapping history can induce dependence. No claim of predictive advantage.'],
              'forecasts': records}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', required=True)
    parser.add_argument('--output', default='reports/experiments/catboost-holdout.json')
    parser.add_argument('--test-size', type=int, default=400)
    parser.add_argument('--validation-size', type=int, default=200)
    parser.add_argument('--iterations', type=int, default=160)
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--through-draw', type=int, help='Freeze historical snapshot at this draw')
    args = parser.parse_args()
    import mini_lotto_predictor as m
    history = m.load_history(args.csv_path)
    if args.through_draw is not None:
        if args.through_draw not in set(history.Numer):
            parser.error('Requested cutoff draw is not in history')
        history = history.loc[history.Numer <= args.through_draw].copy()
    report = benchmark(history, args.output, args.test_size,
                       args.validation_size, args.iterations, args.legacy)
    print(json.dumps(report['results'], indent=2))
