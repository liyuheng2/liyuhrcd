"""Regression checks for the evidence and bounded-channel equations."""
import unittest

import torch

from tmc_acrib import ConditionalChannel, action, counterfactual, reliability, select_alpha
from tmc_crib import discount_evidence
from tmc_hcrd import fuse


class ACRIBMathTest(unittest.TestCase):
    def test_counterfactuals_match_direct_fusion(self):
        evidence = torch.rand((3, 3, 5), generator=torch.Generator().manual_seed(15)) * 20
        mask = torch.tensor([[True, True, True], [True, False, True], [False, True, False]])
        alpha, candidates = counterfactual(evidence, mask)
        torch.testing.assert_close(alpha, fuse(evidence * mask[..., None]))
        for i in range(mask.shape[1]):
            for j, weight in enumerate((.5, 0.)):
                retention = torch.ones_like(mask, dtype=evidence.dtype)
                retention[:, i] = weight
                torch.testing.assert_close(candidates[:, i, j], fuse(evidence * (retention * mask)[..., None]))
                discounted = discount_evidence(evidence, reliability(evidence, retention, mask), mask)
                torch.testing.assert_close(discounted, evidence * (retention * mask)[..., None])

    def test_action_ties_threshold_and_single_modality(self):
        evidence = torch.ones(3, 3, 2)
        mask = torch.tensor([[True, True, False], [False, True, False], [True, False, True]])
        delta = torch.tensor([[[.1, .4], [.4, .05], [.8, .9]], [[.9, .9], [.9, .9], [.9, .9]],
                              [[.2, .05], [.9, .9], [.1, .1]]])
        modality, choice, selected = action(delta, mask, .2)
        self.assertEqual((modality[0].item(), choice[0].item()), (1, 0))
        self.assertEqual(selected.tolist(), [True, False, False])
        alpha, candidates = counterfactual(evidence, mask)
        final, weight = select_alpha(alpha, candidates, delta, mask, .2)
        torch.testing.assert_close(final, fuse(evidence * (weight * mask)[..., None]))

    def test_bounded_channel_closed_form_kl(self):
        torch.manual_seed(22)
        channel = ConditionalChannel(8, 5, 64, 16, 'acrib')
        x, context = torch.randn(6, 3, 8), torch.randn(6, 3, 5)
        mean, sd, cond, kl, capacity = channel.distribution(x, context, 0, None)
        prior_input = torch.cat((torch.zeros_like(x), cond, torch.zeros_like(x[..., :1])), -1)
        prior_mean, prior_logvar = channel.gaussian(channel.encoder(prior_input)).chunk(2, -1)
        prior = torch.distributions.Normal(prior_mean, (.5 * prior_logvar.clamp(-8, 8)).exp())
        expected = torch.distributions.kl_divergence(torch.distributions.Normal(mean, sd), prior).sum(-1)
        torch.testing.assert_close(kl, expected, atol=2e-6, rtol=1e-5)
        self.assertTrue((kl >= 0).all())
        self.assertTrue((kl <= ((torch.e + 2) / 2) * capacity + 1e-6).all())


if __name__ == '__main__':
    unittest.main()
