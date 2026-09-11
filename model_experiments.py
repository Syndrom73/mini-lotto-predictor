"""Chronological ranking/filter audit; test results never select settings."""
import copy
import json
from pathlib import Path
import numpy as np


def probabilities(m, bundle, modern, X, start, stop):
    scaled = bundle.scaler.transform(X[start:stop]).astype(np.float32)
    neural = m.calibrated_nn_probabilities(bundle.calibrator, m.model_logits(bundle.model, scaled))
    matrix = m.history_to_matrix(modern)
    stats = m.statistical_predictions(matrix, m.dataset_offset()+start, m.dataset_offset()+stop)
    result = m.ensemble_probabilities(neural, stats, bundle.ensemble_weight)
    if bundle.hybrid_weight:
        hybrid = m.calibrated_nn_probabilities(bundle.hybrid_calibrator,
                                              m.model_logits(bundle.hybrid_model, scaled))
        result = m.blend_hybrid_probabilities(result, hybrid, bundle.hybrid_weight)
    return result


def summarize(hits):
    a = np.asarray(hits)
    return {"draws": len(a), "any_set_ge3": float((a.max(axis=1)>=3).mean()),
            "set_1_mean_hits": float(a[:,0].mean()), "set_2_mean_hits": float(a[:,1].mean())}


def evaluate_sets(m, modern, scores, start):
    matrix = m.history_to_matrix(modern)
    original = (m.CFG.pair_score_weight, m.CFG.spread_weight)
    output = {}
    try:
        for variant in ("production", "no_rotation", "no_pair_score", "probabilities_only"):
            m.CFG.pair_score_weight = 0 if variant in ("no_pair_score", "probabilities_only") else original[0]
            m.CFG.spread_weight = 0 if variant == "probabilities_only" else original[1]
            previous = None
            hits, random_hits = [], []
            for row, p in enumerate(scores):
                i = m.dataset_offset()+start+row
                draw = int(modern.iloc[i]["Numer"])
                prediction = m.generate_two_sets(p, matrix[:i], draw, modern.iloc[i]["Date"],
                    previous_prediction=None if variant=="no_rotation" else previous)
                actual = set(np.flatnonzero(matrix[i])+1)
                hits.append([len(actual & set(prediction.set_1)), len(actual & set(prediction.set_2))])
                # Matched overlap; many random pairs per draw reduce simulation noise.
                overlap = len(set(prediction.set_1)&set(prediction.set_2))
                rng = np.random.default_rng(m.SEED+draw)
                for _ in range(100):
                    balls = rng.permutation(np.arange(1,43))
                    first = balls[:5]
                    second = np.concatenate((first[:overlap],balls[5:10-overlap]))
                    random_hits.append([len(actual & set(first)),len(actual & set(second))])
                previous = m.StoredPrediction(draw, prediction.next_draw_date,
                                               prediction.set_1, prediction.set_2)
            output[variant] = summarize(hits)
            output[variant]["random_reference"] = summarize(random_hits)
        return output
    finally:
        m.CFG.pair_score_weight, m.CFG.spread_weight = original


def run_comparison(m, history, archive, output_path, evaluation_draws=50):
    """Train two equal-budget variants; retain requested ranking=0.1.

    Validation is also used for calibration, so its scores are exploratory.
    Test blocks are untouched by gradients/calibration. No test-based selection.
    Rotation starts empty at each block boundary and then follows its own sets.
    """
    original = copy.deepcopy(m.CFG)
    modern = m.get_modern_history(history)
    X, Y = m.build_dataset(modern)
    split = m.temporal_split(X,Y)
    report = {"evaluation_draws_per_block": evaluation_draws,
              "note": "Exploratory validation; untouched chronological test. No parameter selection from test.",
              "variants": {}}
    chosen = None
    try:
        for weight in (0.0, original.ranking_loss_weight):
            m.CFG.ranking_loss_weight = weight
            bundle = m.train_predictor(history, archive, finalize=False)
            results = {}
            for name, low, stop in (("validation", split.val_start_dataset, split.test_start_dataset),
                                    ("test", split.test_start_dataset, len(X))):
                start = max(low, stop-evaluation_draws)
                scores = probabilities(m,bundle,modern,X,start,stop)
                results[name] = {"first_draw": int(modern.iloc[m.dataset_offset()+start]["Numer"]),
                                 "last_draw": int(modern.iloc[m.dataset_offset()+stop-1]["Numer"]),
                                 "brier": float(np.mean((scores-Y[start:stop])**2)),
                                 "sets": evaluate_sets(m,modern,scores,start)}
                print(f"Comparison ranking={weight}: {name} finished", flush=True)
            report["variants"][str(weight)] = results
            if weight == original.ranking_loss_weight:
                chosen = bundle
    finally:
        for key,value in m.asdict(original).items():
            setattr(m.CFG,key,value)
    path = Path(output_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    m.online_update(chosen,history,archive)
    return chosen
