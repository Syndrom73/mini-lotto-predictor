import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import catboost_ranker as r
import mini_lotto_predictor as m
import ranker_pipeline as pipeline
from ranker_backtest import choose, summary


class RankerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.y = np.zeros((400,42), dtype=np.float32)
        for row in self.y:
            row[rng.choice(42,5,replace=False)] = 1

    def test_features_do_not_see_target_or_future(self):
        other = self.y.copy()
        other[300:] = other[300:,::-1]
        np.testing.assert_array_equal(r.features(self.y)[:301],r.features(other)[:301])
        np.testing.assert_array_equal(r.features(self.y[:300])[-1],r.features(self.y)[300])

    def test_holdout_does_not_select_model_or_calibrator(self):
        x = r.features(self.y)
        numbers = np.arange(129,529)
        a, ca = choose(x, self.y, numbers, 380, validation=20, iterations=2)
        other = self.y.copy(); other[380:] = other[380:,::-1]
        b, cb = choose(r.features(other), other, numbers, 380, validation=20, iterations=2)
        self.assertEqual(a[1:], b[1:])
        self.assertEqual([c['validation'] for c in ca],[c['validation'] for c in cb])
        np.testing.assert_allclose(r.scores(a[0],x[380:381]), r.scores(b[0],x[380:381]))

    def test_native_save_roundtrip_and_probabilities(self):
        x = r.features(self.y)
        model = r.fit(x[250:270], self.y[250:270], r.Settings(iterations=3))
        raw = r.scores(model, x[270:280])
        cal = r.calibrate(raw,self.y[270:280])
        p = r.probabilities(raw,cal)
        self.assertTrue(((p>0)&(p<1)).all())
        np.testing.assert_allclose(p.sum(1),5)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.zip'
            r.save(model, {'version':1,'calibration':cal}, path)
            restored,metadata=r.load(path)
            np.testing.assert_allclose(r.scores(restored,x[270:280]),raw)
            self.assertEqual(metadata['calibration'],cal)

    def test_selection_rotation_disjointness_and_reproducibility(self):
        for strategy in ('greedy','balanced'):
            previous=None
            for draw in range(100):
                tickets=r.select_two(np.ones(42),previous,strategy,draw)
                self.assertEqual(tickets,r.select_two(np.ones(42),previous,strategy,draw))
                self.assertEqual(len(set(tickets[0])|set(tickets[1])),10)
                self.assertTrue(all(len(t)==5 and all(type(n) is int and 1<=n<=42 for n in t) for t in tickets))
                if previous:
                    self.assertTrue(all(len(set(t)&set(old))<=3 for t in tickets for old in previous))
                previous=tickets

    def history(self):
        dates=pd.date_range('2020-01-01',periods=len(self.y))
        return m.prepare_history(pd.DataFrame([[129+i,d.day,d.month,d.year,*list(np.flatnonzero(y)+1)]
            for i,(d,y) in enumerate(zip(dates,self.y))],columns=m.REQUIRED_COLUMNS))

    def test_daily_report_and_idempotent_pending_forecast(self):
        history=self.history()
        x=r.features(self.y)
        model=r.fit(x[250:270],self.y[250:270],r.Settings(iterations=2))
        metadata={'version':1,'settings':r.asdict(r.Settings(iterations=2)),
                  'calibration':[1.,-2.], 'history_length':399,'history_sha256':r.digest(self.y[:399]),
                  'source_draw':527,'trained_through':398,'generation':'test'}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); modelpath=root/'model.zip'; r.save(model,metadata,modelpath)
            history[m.REQUIRED_COLUMNS].to_csv(root/'history.csv',index=False)
            state=root/'last_prediction.json'; report=root/'latest.md'
            prior=m.Prediction(528,history.iloc[-1].Date,(1,2,3,4,5),(6,7,8,9,10),0,0,[])
            m.save_prediction_state(prior,state)
            args=dict(csv_path=str(root/'history.csv'),bundle_path=str(modelpath),mode='daily',
                      state_path=str(state),report_path=str(report))
            pred=pipeline.main(**args)
            text=report.read_text()
            for token in ['PO LOSOWANIU MINI LOTTO 528','DATA LOSOWANIA:', 'WYLOSOWANE LICZBY:',
                          'OCENA POPRZEDNIEJ PROGNOZY','trafienia:', 'PROGNOZA NASTĘPNEGO LOSOWANIA MINI LOTTO 529',
                          'PRZEWIDYWANA DATA:', 'ZESTAW 1:', 'ZESTAW 2:', 'CatBoost Ranker']:
                self.assertIn(token,text)
            self.assertNotIn('Temporalny CNN',text)
            before={p.name:p.read_bytes() for p in root.iterdir()}
            with patch.object(r,'load',side_effect=AssertionError('pending must not load model')):
                repeated=pipeline.main(**args)
            self.assertEqual(pred.set_1,repeated.set_1)
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})
            archive=json.loads((root/'probability_history.json').read_text())
            self.assertEqual(archive['529']['source_draw_number'],528)

    def test_training_preserves_pending_forecast(self):
        history=self.history()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            history[m.REQUIRED_COLUMNS].to_csv(root/'history.csv',index=False)
            state=root/'state.json'; report=root/'latest.md'
            prior=m.Prediction(529,history.iloc[-1].Date+pd.Timedelta(days=1),
                              (1,2,3,4,5),(6,7,8,9,10),0,0,[])
            m.save_prediction_state(prior,state); report.write_text('original report')
            before=(state.read_bytes(),report.read_bytes())
            with patch.object(pipeline,'train',return_value=(None,{'candidates':[]})) as train:
                result=pipeline.main(str(root/'history.csv'),bundle_path=str(root/'model.zip'),
                                     mode='train',state_path=str(state),report_path=str(report))
                train.assert_called_once()
            self.assertEqual(result.set_1,prior.set_1)
            self.assertEqual(before,(state.read_bytes(),report.read_bytes()))

    def test_history_rollback_is_rejected(self):
        history=self.history()
        x=r.features(self.y)
        model=r.fit(x[250:270],self.y[250:270],r.Settings(iterations=2))
        metadata={'version':1,'history_length':401,'history_sha256':'invalid'}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); r.save(model,metadata,root/'model.zip')
            history[m.REQUIRED_COLUMNS].to_csv(root/'history.csv',index=False)
            with self.assertRaisesRegex(ValueError,'rolled back'):
                pipeline.main(str(root/'history.csv'),bundle_path=str(root/'model.zip'),mode='daily')

    def test_hit_metric_counts_either_ticket(self):
        result=summary([[0,3],[4,0],[2,2],[0,0]])
        self.assertEqual(result['any_set_ge3_count'],2)
        self.assertEqual(result['any_set_ge3'],.5)

if __name__=='__main__': unittest.main()
