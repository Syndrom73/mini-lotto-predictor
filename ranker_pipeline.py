"""CatBoost backend using the existing report, state and audit contracts."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json
import os

import numpy as np
import pandas as pd
import catboost_ranker as r
from ranker_backtest import choose


def train(history, output_path):
    import mini_lotto_predictor as m
    modern = m.get_modern_history(history)
    y = m.history_to_matrix(modern)
    x = r.features(y)
    (model, calibration, settings), candidates = choose(x, y, modern.Numer.to_numpy(), len(y))
    # Keep the exact calibrated model. The newest draws update causal features;
    # they are not falsely advertised as gradient-training examples.
    metadata = {'version': r.VERSION, 'settings': r.asdict(settings),
                'calibration': calibration, 'history_length': len(y),
                'history_sha256': r.digest(y),
                'source_draw': int(modern.iloc[-1].Numer),
                'trained_through': int(modern.iloc[-201].Numer),
                'generation': datetime.now(timezone.utc).isoformat(),
                'candidates': candidates}
    r.save(model, metadata, output_path)
    return model, metadata


def main(csv_path=None, next_draw_date=None, bundle_path='mini_lotto_catboost.zip',
         mode='train', state_path=None, report_path=None, prediction_history_path=None):
    import mini_lotto_predictor as m
    if mode not in ('daily', 'train'):
        raise ValueError('Unknown mode')
    history = m.load_history(m.resolve_csv_path(csv_path))
    modern = m.get_modern_history(history)
    matrix = m.history_to_matrix(modern)
    latest = int(modern.iloc[-1].Numer)
    state_file = Path(state_path) if state_path else m.prediction_state_path(bundle_path)
    previous = m.load_prediction_state(state_file)
    pending = previous is not None and previous.draw_number > latest
    if pending and mode == 'daily':
        return m.Prediction(previous.draw_number, previous.draw_date, previous.set_1,
                            previous.set_2, float('nan'), float('nan'), [])
    if mode == 'train':
        model, metadata = train(history, bundle_path)
        comparison = state_file.with_name('model_comparison.json')
        comparison.parent.mkdir(parents=True, exist_ok=True)
        temporary = comparison.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'backend': 'catboost', 'selection': metadata['candidates'],
                                        'note': 'Temporal validation only; independent holdout is a separate artifact.'},
                                       indent=2), encoding='utf-8')
        os.replace(temporary, comparison)
    else:
        model, metadata = r.load(bundle_path)
        length = metadata['history_length']
        if len(matrix) < length or r.digest(matrix[:length]) != metadata['history_sha256']:
            raise ValueError('History changed or rolled back; run train before prediction')
    if pending:
        return m.Prediction(previous.draw_number, previous.draw_date, previous.set_1,
                            previous.set_2, float('nan'), float('nan'), [])
    date = pd.Timestamp(next_draw_date) if next_draw_date else m.infer_next_draw_date(modern)
    if pd.isna(date) or date <= modern.iloc[-1].Date:
        raise ValueError('Next draw date must follow the latest result')
    p = r.probabilities(r.scores(model, r.features(matrix)[-1:]), metadata['calibration'])[0]
    old_sets = (previous.set_1, previous.set_2) if previous else None
    first, second = r.select_two(p, old_sets, metadata['settings']['selector'], r.SEED+latest+1)
    prediction = m.Prediction(latest+1, date, first, second, float(p[np.array(first)-1].sum()),
                             float(p[np.array(second)-1].sum()),
                             sorted([(i+1,float(v)) for i,v in enumerate(p)], key=lambda t:-t[1]))
    prediction.backend = 'catboost'
    report = m.print_complete_draw_summary(history, previous, prediction)
    archive_path = state_file.with_name('probability_history.json')
    archive = m.audit_probability_archive(archive_path, history)
    report += (f"\nMODEL: CatBoost Ranker; kara {metadata['settings']['loss_function']}; "
               f"wybór zestawów: {metadata['settings']['selector']}.\n"
               f"Uczenie drzew do losowania {metadata['trained_through']}; "
               f"kalibracja i walidacja do {metadata['source_draw']}; "
               f"cechy aktualne do {latest}.\n"
               "Tryb daily aktualizuje cechy bez treningu; pełny trening według harmonogramu.\n"
               "Niezależny test historyczny: reports/experiments/catboost-holdout.json; "
               "wyniki testu nie dobierają parametrów modelu.\n"
               "Wyniki gry są losowe; predykcja nie gwarantuje wygranej.\n")
    evaluations = [v['evaluation'] for v in archive.values() if 'evaluation' in v]
    if evaluations:
        report += (f"OCENA ZAPISANYCH PROGNOZ (wszystkie wersje modeli): {len(evaluations)} losowań; "
                   f"średni Brier {np.mean([e['brier'] for e in evaluations]):.6f}; "
                   f"punkt odniesienia {(5/42)*(37/42):.6f}.\n"
                   f"Co najmniej jeden zestaw z >=3/5: "
                   f"{np.mean([max(e['set_1_hits'],e['set_2_hits'])>=3 for e in evaluations]):.2%}.\n")
    bundle_info = SimpleNamespace(modern_history=modern, model_revision=0,
                                 model_generation='catboost:'+metadata['generation'],
                                 learned_through=metadata['trained_through'])
    m.save_probability_archive(archive_path, archive, prediction, bundle_info)
    m.update_prediction_history(Path(prediction_history_path) if prediction_history_path else
                                state_file.with_name('prediction_history.csv'), history, previous, prediction)
    m.save_prediction_state(prediction, state_file)
    if report_path:
        dest = Path(report_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(dest.suffix+'.tmp')
        temp.write_text(report.lstrip(), encoding='utf-8')
        os.replace(temp,dest)
    return prediction
