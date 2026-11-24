import unittest

import torch

from tmc_cibf import TMC_CIBF, conjunctive, fuse_evidence, predict


class CIBFTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = TMC_CIBF([2, 3, 4], 2)
        self.inputs = [torch.randn(4, dim) for dim in (2, 3, 4)]
        self.mask = torch.tensor([[1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 0]]).bool()

    def test_conflict_mass_and_ds_ordering(self):
        belief = torch.tensor([[[.9, 0], [0, .9]]])
        uncertainty = torch.full((1, 2, 1), .1)
        out = conjunctive(belief, uncertainty)
        ds = conjunctive(belief, uncertainty, ds=True)
        torch.testing.assert_close(out['u'], torch.tensor([.82]))
        torch.testing.assert_close(ds['u'], torch.tensor([.01 / .19]))
        torch.testing.assert_close(out['conflict'], torch.tensor([.81]))
        torch.testing.assert_close(out['p'], conjunctive(belief.flip(1), uncertainty)['p'])
        evidence = torch.rand(12, 3, 4) * 8
        retention = torch.rand(12, 3)
        conservative = fuse_evidence(evidence, retention)
        normalized = fuse_evidence(evidence, retention, ds=True)
        torch.testing.assert_close(conservative['p'].argmax(1), normalized['p'].argmax(1))

    def test_shapley_weights_and_empty_input(self):
        model = self.model.eval()
        out = model(self.inputs, self.mask)
        edge = out['contribution']
        torch.testing.assert_close(edge['weight'] @ model.identity, self.mask.float())
        torch.testing.assert_close(edge['valid'].sum(1), torch.tensor([12, 4, 1, 0]))
        torch.testing.assert_close(edge['current'].float() @ model.identity, self.mask.float())
        result = predict(model, self.inputs, self.mask)
        torch.testing.assert_close(result['probability'].sum(1), torch.ones(4))
        torch.testing.assert_close(result['probability'][-1], torch.full((2,), .5))
        torch.testing.assert_close(result['uncertainty'][-1], torch.tensor(1.))
        self.assertEqual(result['prediction'][-1].item(), -1)
        perturbed = [x.masked_fill(~self.mask[:, i, None], 99) for i, x in enumerate(self.inputs)]
        torch.testing.assert_close(predict(model, perturbed, self.mask)['probability'], result['probability'])

    def test_gradient_boundaries(self):
        out = self.model(self.inputs, self.mask)
        labels = torch.tensor([0, 1, 0, 1])
        _, (task, ib, contribution) = self.model.loss(out, labels, labels[:, None].expand_as(self.mask),
                                                       self.mask, .5, .001, .5, False)
        smce = tuple(self.model.smce.parameters())
        encoder = tuple(self.model.project.parameters())
        self.assertTrue(all(g is None for g in torch.autograd.grad(task + ib, smce, allow_unused=True)))
        self.assertTrue(all(g is None for g in torch.autograd.grad(contribution, encoder, retain_graph=True,
                                                                   allow_unused=True)))
        self.assertTrue(any(g.abs().sum() > 0 for g in torch.autograd.grad(contribution, smce)))


if __name__ == '__main__':
    unittest.main()
