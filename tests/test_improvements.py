import copy
import unittest
from itertools import combinations
from unittest.mock import patch
import numpy as np
import torch
import mini_lotto_predictor as m


class SearchTests(unittest.TestCase):
    def test_vectorized_search_matches_exhaustive(self):
        old=copy.deepcopy(m.CFG)
        try:
            m.CFG.n_numbers=8
            m.CFG.draw_size=3
            rng=np.random.default_rng(123)
            p=rng.random(8); pairs=rng.random((8,8))
            constraints=[((1,2,3),1)]
            actual=m.exact_best_set(p,pairs,overlap_constraints=constraints)
            legal=[s for s in combinations(range(1,9),3) if len(set(s)&{1,2,3})<=1]
            expected=max(legal,key=lambda s:m.set_fitness(s,p,pairs))
            self.assertEqual(actual[0],expected)
            self.assertAlmostEqual(actual[1],m.set_fitness(expected,p,pairs))
        finally:
            for k,v in m.asdict(old).items(): setattr(m.CFG,k,v)


class GuardTests(unittest.TestCase):
    def test_rejects_degraded_update(self):
        # Exercise the actual guard with a small deterministic feature fixture.
        from types import SimpleNamespace
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        old=copy.deepcopy(m.CFG)
        try:
            m.CFG.device='cpu'
            model=torch.nn.Linear(2,42)
            before=copy.deepcopy(model.state_dict())
            X=np.zeros((66,2),dtype=np.float32)
            Y=np.zeros((66,42),dtype=np.float32); Y[:,:5]=1
            history=pd.DataFrame({'Numer':np.arange(66)})
            bundle=SimpleNamespace(model=model,hybrid_model=None,scaler=StandardScaler().fit(X),
                                   learned_through=64,model_revision=0)
            original=m.feedback_loss
            calls=[0]
            def loss(x,y,w):
                calls[0]+=1
                if calls[0]==1: return x.sum()*0+1
                if calls[0]==4: return x.sum()*0+2
                return original(x,y,w)
            with patch.object(m,'get_modern_history',return_value=history), \
                 patch.object(m,'build_dataset',return_value=(X,Y)), \
                 patch.object(m,'dataset_offset',return_value=0), \
                 patch.object(m,'refresh_bundle_history'), patch.object(m,'feedback_loss',side_effect=loss):
                m.online_update(bundle,history)
            for key in before: self.assertTrue(torch.equal(before[key],model.state_dict()[key]))
            self.assertEqual(bundle.learned_through,65)
        finally:
            for k,v in m.asdict(old).items(): setattr(m.CFG,k,v)

if __name__=='__main__': unittest.main()
