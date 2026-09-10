import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import mini_lotto_predictor as m


class OnlineLearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        m.set_seed()
        self.old = m.asdict(m.CFG)
        m.CFG.sequence_length = 2
        m.CFG.rolling_windows = (2,)
        m.CFG.pair_window = 2
        m.CFG.hidden_1, m.CFG.hidden_2, m.CFG.hidden_3 = 12, 8, 4
        m.CFG.cnn_channels_1, m.CFG.cnn_channels_2 = 4, 4
        m.CFG.hybrid_context_hidden = 8
        m.CFG.device = 'cpu'
        rng = np.random.default_rng(1)
        rows = []
        for i in range(12):
            date = pd.Timestamp('2020-01-01') + pd.Timedelta(days=i)
            rows.append([129+i, date.day, date.month, date.year,
                         *rng.choice(np.arange(1,43), 5, replace=False)])
        self.history = m.prepare_history(pd.DataFrame(rows, columns=m.REQUIRED_COLUMNS))
        X, _ = m.build_dataset(self.history)
        self.bundle = m.PredictorBundle(
            m.MiniLottoMLP(X.shape[1]), StandardScaler().fit(X[:5]),
            None, .5, None, None, None, self.history.iloc[:-1].copy(),
            m.history_to_matrix(self.history.iloc[:-1]), X.shape[1],
            hybrid_model=m.MiniLottoTemporalHybrid(X.shape[1]), learned_through=139)

    def tearDown(self):
        for key, value in self.old.items():
            setattr(m.CFG, key, value)

    def test_update_both_networks_and_idempotent_roundtrip(self):
        before = [p.detach().clone() for p in self.bundle.model.parameters()]
        cnn_before = [p.detach().clone() for p in self.bundle.hybrid_model.parameters()]
        self.assertEqual(m.online_update(self.bundle, self.history), 1)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,self.bundle.model.parameters())))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(cnn_before,self.bundle.hybrid_model.parameters())))
        with tempfile.TemporaryDirectory() as tmp:
            file = str(Path(tmp)/'model.pt')
            m.save_bundle(self.bundle, file)
            restored = m.load_bundle(file)
            state = {k:v.clone() for k,v in restored.model.state_dict().items()}
            self.assertEqual(m.online_update(restored, self.history), 0)
            self.assertEqual(restored.learned_through, 140)
            self.assertTrue(all(torch.equal(v,restored.model.state_dict()[k]) for k,v in state.items()))

    def test_archive_scores_saved_sets_and_preserves_forecast(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'history.json'
            actual = m._draw_numbers(self.history.iloc[-1])
            pred = m.Prediction(140, self.history.iloc[-1].Date,
                tuple(actual), (1,2,3,4,5), 0.,0., [(i,5/42) for i in range(1,43)])
            m.save_probability_archive(path, {}, pred, self.bundle)
            archive = m.audit_probability_archive(path,self.history)
            self.assertEqual(archive['140']['evaluation']['set_1_hits'], 5)
            self.assertAlmostEqual(archive['140']['evaluation']['brier'], (5/42)*(37/42))
            pred.set_1 = (1,2,3,4,5)
            m.save_probability_archive(path, archive, pred, self.bundle)
            self.assertEqual(m.audit_probability_archive(path,self.history)['140']['set_1'],actual)

    def test_full_training_catches_up_after_untouched_test(self):
        m.CFG.validation_size = m.CFG.test_size = 2
        m.CFG.batch_size = 2
        m.CFG.epochs = 1
        bundle = m.train_predictor(self.history)
        self.assertEqual(bundle.learned_through, 140)
        self.assertEqual(bundle.model_revision, 4)
        self.assertTrue(np.isfinite(bundle.test_metrics.brier))
        self.assertEqual(m.online_update(bundle, self.history), 0)

    def test_future_results_do_not_affect_earlier_update(self):
        other = m.copy.deepcopy(self.bundle)
        self.bundle.learned_through = other.learned_through = 138
        m.online_update(self.bundle, self.history.iloc[:-1])
        changed = self.history.copy()
        changed.loc[changed.index[-1], m.NUMBER_COLUMNS] = [1,2,3,4,5]
        # Updating through 139 cannot see the target at 140.
        m.online_update(other, changed.iloc[:-1])
        self.assertTrue(all(torch.equal(v,other.model.state_dict()[k])
                            for k,v in self.bundle.model.state_dict().items()))




class FeedbackTests(unittest.TestCase):
    def test_bounds_and_future_rejection(self):
        self.assertEqual(m.feedback_weight(None, 140), 1)
        record = {'source_draw_number':139, 'evaluation':{
            'brier':1., 'set_1_hits':0, 'set_2_hits':0}}
        self.assertEqual(m.feedback_weight(record,140),1.5)
        record['source_draw_number']=140
        with self.assertRaises(ValueError):
            m.feedback_weight(record,140)

    def test_brier_loss_is_finite_and_weights_change_gradient(self):
        logits = torch.zeros((2,42), requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[0,:5]=1; targets[1,5:10]=1
        a=m.feedback_loss(logits, targets, torch.ones(2))
        ga=torch.autograd.grad(a,logits)[0]
        b=m.feedback_loss(logits, targets, torch.tensor([1.5,1.]))
        gb=torch.autograd.grad(b,logits)[0]
        self.assertTrue(torch.isfinite(b))
        self.assertFalse(torch.equal(ga,gb))

    def test_two_days_across_month_boundary(self):
        from automation.training_schedule import training_due
        from datetime import date
        self.assertTrue(training_due(date(2026,9,10),None))
        self.assertFalse(training_due(date(2026,10,1),'2026-09-30'))
        self.assertTrue(training_due(date(2026,10,2),'2026-09-30'))

if __name__ == '__main__':
    unittest.main()
