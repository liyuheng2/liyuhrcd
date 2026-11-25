"""Task-supervised risk estimates and conservative fusion for EC-RIB."""
import torch
from torch import nn
from torch.nn import functional as F

from model import EPS, HCRDRIB, entropy, jsd, masked_mean


MODES = ('task_uniform', 'task_fixed', 'task_dynamic', 'task_shuffled')


class TaskRisk(HCRDRIB):
    def __init__(self, dims, classes, mode, width, latent):
        super().__init__(dims, classes, 'fixed_ib', width, latent)
        self.method = mode
        self.pair_detector = nn.Linear(width, 1)

    def forward(self, inputs, mask, uniform=False):
        z, hidden, p = self.encode(inputs)
        uniform_alpha = mask / mask.sum(1, keepdim=True)
        if uniform:
            return self.fuse(z, p, uniform_alpha, p.new_ones(len(mask)))

        p_detached = p.detach()
        modal = torch.cat((self.observation(hidden.detach()), entropy(p_detached)[..., None],
                           p_detached.amax(-1, keepdim=True)), -1)
        damage, kl_o = self.predict_risk(modal, self.identity.weight, 0)
        # The original A bottleneck uses tanh. Map it to a nonnegative risk.
        difficulty, kl_a = self.predict_risk(
            torch.cat((modal, damage.detach()[..., None]), -1), self.identity.weight, 1)
        difficulty = (1 + difficulty) / 2
        left, right = self.left, self.right
        pair = [jsd(p_detached[:, left], p_detached[:, right])]
        for risk in (damage.detach(), difficulty.detach()):
            pair.extend(((risk[:, left] + risk[:, right]) / 2,
                         (risk[:, left] - risk[:, right]).abs()))
        pair_inputs = torch.cat(((z.detach()[:, left] - z.detach()[:, right]).abs(),
                                 torch.stack(pair, -1)), -1)
        pair_id = (self.identity(left) + self.identity(right)) / 2
        harm, kl_c = self.predict_risk(pair_inputs, pair_id, 2)
        swap_logit = self.pair_detector(self.risk_input[2](pair_inputs)).squeeze(-1)
        pair_mask = mask[:, left] & mask[:, right]

        # Split pair risk in proportion to each endpoint's estimated task risk.
        risk = 0.5 * (damage + difficulty)
        share = (risk[:, left] + EPS) / (risk[:, left] + risk[:, right] + 2 * EPS)
        harm = harm * pair_mask
        degree = pair_mask.to(p.dtype) @ (self.incidence_l + self.incidence_r)
        attributed_harm = ((harm * share) @ self.incidence_l
                           + (harm * (1 - share)) @ self.incidence_r) / degree.clamp_min(1)
        utility = -(risk + attributed_harm)
        weighted = (utility / 0.25).masked_fill(~mask, -torch.inf).softmax(1)
        fraction = 0.0 if self.method == 'task_uniform' else 0.25
        alpha = (1 - fraction) * uniform_alpha + fraction * weighted
        result = self.fuse(z, p, alpha.detach(), p.new_ones(len(mask)))
        result.update(score=result['p'].amax(-1), risks=(damage, difficulty, harm),
                      kl=(kl_o, kl_a, kl_c), masks=(mask, mask, pair_mask), swap_logit=swap_logit)
        return result

    @torch.no_grad()
    def targets(self, clean_p, perturbed_p, labels):
        take = labels[:, None, None].expand(-1, clean_p.size(1), 1)
        clean = clean_p.gather(-1, take).squeeze(-1)
        noisy = perturbed_p.gather(-1, take).squeeze(-1)
        # 1-exp(-positive NLL increase) avoids an unbounded regression target.
        damage = 1 - (noisy / clean.clamp_min(EPS)).clamp_max(1)
        left, right = self.left, self.right
        clean_pair = (clean[:, left] + clean[:, right]) / 2
        noisy_pair = (noisy[:, left] + noisy[:, right]) / 2
        harm = 1 - (noisy_pair / clean_pair.clamp_min(EPS)).clamp_max(1)
        return damage, 1 - clean, harm

    def supervised_risk(self, result, targets, swap_pairs, rng):
        losses, rates = [], []
        for prediction, target, kl, mask in zip(result['risks'], targets, result['kl'], result['masks']):
            weights = torch.ones_like(target)
            if self.method in {'task_dynamic', 'task_shuffled'}:
                weights = (1 + target).reciprocal()
                if self.method == 'task_shuffled':
                    valid = weights[mask]
                    order = torch.randperm(len(valid), generator=rng).to(valid.device)
                    weights[mask] = valid[order]
                mean_weight = masked_mean(weights, mask)
                weights = weights / mean_weight.clamp_min(EPS)
            losses.append(masked_mean((prediction - target).square(), mask))
            rates.append(masked_mean(kl * weights, mask))
        swap_loss = F.binary_cross_entropy_with_logits(
            result['swap_logit'], swap_pairs.to(result['swap_logit'].dtype), reduction='none')
        return torch.stack(rates).mean(), torch.stack(losses).mean() + 0.25 * masked_mean(swap_loss, result['masks'][2])
