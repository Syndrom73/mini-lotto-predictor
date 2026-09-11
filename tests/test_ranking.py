import unittest
import torch
import mini_lotto_predictor as m

class RankingTests(unittest.TestCase):
    def test_direction_and_step(self):
        y = torch.tensor([[1., 0., 0.]])
        x = torch.tensor([[-1., 1., 2.]], requires_grad=True)
        loss = m.pairwise_ranking_loss(x, y).mean()
        loss.backward()
        self.assertLess(x.grad[0, 0].item(), 0)
        self.assertTrue((x.grad[0, 1:] > 0).all())
        self.assertLess(m.pairwise_ranking_loss(x - .1*x.grad, y).item(), loss.item())

    def test_invariance_and_extremes(self):
        y = torch.tensor([[1., 0., 1., 0.]])
        x = torch.tensor([[1000., -1000., -1000., 1000.]], requires_grad=True)
        a = m.pairwise_ranking_loss(x, y)
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.allclose(a, m.pairwise_ranking_loss(x+10, y)))
        order = [2, 0, 3, 1]
        self.assertTrue(torch.allclose(a, m.pairwise_ranking_loss(x[:,order], y[:,order])))
        a.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_no_pairs_and_batch_independence(self):
        x = torch.randn(2, 42, requires_grad=True)
        self.assertEqual(m.pairwise_ranking_loss(x, torch.zeros_like(x)).sum().item(), 0)
        y = torch.zeros_like(x); y[:, :5] = 1
        self.assertTrue(torch.allclose(m.pairwise_ranking_loss(x,y)[0],
                                      m.pairwise_ranking_loss(x[:1],y[:1])[0]))

    def test_both_training_paths(self):
        x = torch.zeros(2, 42, requires_grad=True)
        y = torch.zeros_like(x); y[:, :5] = 1
        old = m.CFG.ranking_loss_weight
        try:
            m.CFG.ranking_loss_weight = 0
            base = m.WeightedBCE()(x,y)
            feedback = m.feedback_loss(x,y,torch.ones(2))
            m.CFG.ranking_loss_weight = .1
            expected = .1*m.pairwise_ranking_loss(x,y).mean()
            self.assertTrue(torch.allclose(m.WeightedBCE()(x,y)-base,expected))
            self.assertTrue(torch.allclose(m.feedback_loss(x,y,torch.ones(2))-feedback,expected))
        finally:
            m.CFG.ranking_loss_weight = old

if __name__ == '__main__':
    unittest.main()
