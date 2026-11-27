"""Author-method adaptations using the same frozen features and availability masks."""
import torch
from torch import nn
from torch.nn import functional as F

from model import EPS, masked_mean


BASELINES = {'late', 'concat', 'gate', 'tmc', 'etmc', 'qmf', 'mmdynamics',
             'cdpr_features', 'sed_ud_features'}


def evidential_loss(alpha, labels, anneal):
    onehot = F.one_hot(labels, alpha.shape[-1]).float()
    strength = alpha.sum(-1, keepdim=True)
    expected_ce = ((strength.digamma() - alpha.digamma()) * onehot).sum(-1)
    adjusted = (alpha - 1) * (1 - onehot) + 1
    mass = adjusted.sum(-1)
    kl = (mass.lgamma() - adjusted.lgamma().sum(-1) - mass.new_tensor(alpha.shape[-1]).lgamma()
          + ((adjusted - 1) * (adjusted.digamma() - mass.digamma()[:, None])).sum(-1))
    return expected_ce + anneal * kl


class Baseline(nn.Module):
    def __init__(self, dims, classes, name, rows, width=128):
        super().__init__()
        self.name = name
        self.encoders = nn.ModuleList(nn.Sequential(nn.Linear(dim, width), nn.GELU()) for dim in dims)
        self.heads = nn.ModuleList(nn.Linear(width, classes) for _ in dims)
        if name in {'etmc', 'concat', 'mmdynamics'}:
            self.joint = nn.Sequential(nn.Linear(len(dims) * width, width), nn.GELU(), nn.Linear(width, classes))
        if name == 'gate':
            self.gate = nn.Linear(width, 1)
        if name == 'qmf':
            self.register_buffer('history', torch.zeros(rows, len(dims)))
        if name == 'mmdynamics':
            self.feature_gates = nn.ModuleList(nn.Linear(dim, dim) for dim in dims)
            self.confidence_heads = nn.ModuleList(nn.Linear(width, 1) for _ in dims)

    def forward(self, inputs, mask):
        features = inputs
        if self.name == 'mmdynamics':
            feature_weight = [layer(x).sigmoid() for layer, x in zip(self.feature_gates, inputs)]
            features = [x * w for x, w in zip(inputs, feature_weight)]
        z = torch.stack([layer(x) for layer, x in zip(self.encoders, features)], dim=1)
        logits = torch.stack([layer(z[:, i]) for i, layer in enumerate(self.heads)], dim=1)
        p = logits.softmax(-1)
        alpha = mask / mask.sum(1, keepdim=True)
        extra = {}
        if self.name in {'tmc', 'etmc'}:
            evidence = F.softplus(logits) * mask[..., None]
            per_modal = evidence.unbind(1)
            if self.name == 'etmc':
                pseudo = F.softplus(self.joint((z * mask[..., None]).flatten(1)))
                # Joint evidence is active only when at least two real modalities are available.
                pseudo = pseudo * (mask.sum(1, keepdim=True) > 1)
                per_modal += (pseudo,)
                extra['pseudo'] = pseudo + 1
            combined = torch.zeros_like(per_modal[0])
            for e in per_modal:
                # Algebraic form of Dempster-Shafer evidence combination with uniform base rates.
                combined = combined + e + combined * e / logits.shape[-1]
            joint = combined + 1
            fused = joint / joint.sum(-1, keepdim=True)
            vacuity = logits.shape[-1] / joint.sum(-1)
            mass = evidence.sum(-1)
            alpha = torch.where(mass.sum(1, keepdim=True) > EPS,
                                mass / mass.sum(1, keepdim=True).clamp_min(EPS), alpha)
            extra.update(dirichlet=evidence + 1, joint_dirichlet=joint, score=1 - vacuity)
        elif self.name == 'qmf':
            quality = logits.logsumexp(-1) / 10
            weights = (mask * quality.detach())
            # Use the original additive logit rule for complete inputs. Missing views contribute zero.
            fused = (logits * weights[..., None]).sum(1).softmax(-1)
            extra['quality'] = quality
            alpha = weights.clamp_min(0)
            alpha = torch.where(alpha.sum(1, keepdim=True) > EPS,
                                alpha / alpha.sum(1, keepdim=True).clamp_min(EPS),
                                mask / mask.sum(1, keepdim=True))
        elif self.name == 'mmdynamics':
            confidence = torch.cat([layer(z[:, i]) for i, layer in enumerate(self.confidence_heads)], 1)
            fused = self.joint((z * (confidence * mask)[..., None]).flatten(1)).softmax(-1)
            extra.update(confidence=confidence, feature_weight=feature_weight)
            weights = confidence.clamp_min(0) * mask
            alpha = torch.where(weights.sum(1, keepdim=True) > EPS,
                                weights / weights.sum(1, keepdim=True).clamp_min(EPS), alpha)
        elif self.name == 'concat':
            fused = self.joint((z * mask[..., None]).flatten(1)).softmax(-1)
        else:
            if self.name == 'gate':
                alpha = self.gate(z).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(1)
            fused = (p * alpha[..., None]).sum(1)
        result = {'p': fused, 'single': p, 'alpha': alpha, 'score': fused.max(-1).values, 'logits': logits}
        result.update(extra)
        return result

    def loss(self, result, labels, mask, rows, epoch):
        p = result['p']
        targets = labels[:, None, None].expand(-1, mask.shape[1], 1)
        unimodal_nll = -result['logits'].log_softmax(-1).gather(-1, targets).squeeze(-1)
        fused_nll = -p.clamp_min(EPS).log().gather(1, labels[:, None]).mean()
        if self.name in {'tmc', 'etmc'}:
            anneal = min(1.0, (epoch + 1) / 10)
            loss = evidential_loss(result['joint_dirichlet'], labels, anneal).mean()
            for i, available in enumerate(mask.T):
                loss = loss + masked_mean(evidential_loss(result['dirichlet'][:, i], labels, anneal), available)
            if self.name == 'etmc':
                loss = loss + masked_mean(evidential_loss(result['pseudo'], labels, anneal), mask.sum(1) > 1)
            return loss
        loss = fused_nll + (unimodal_nll * mask).sum(1).div(mask.sum(1)).mean()
        if self.name == 'mmdynamics':
            target_p = result['single'].gather(-1, targets).squeeze(-1)
            for i, weight in enumerate(result['feature_weight']):
                cost = (result['confidence'][:, i] - target_p[:, i]).square() + weight.mean(-1)
                loss = loss + masked_mean(cost, mask[:, i])
        if self.name == 'qmf':
            self.history[rows] += (unimodal_nll.detach() * mask)
            history = self.history[rows]
            span = (self.history.amax(0) - self.history.amin(0)).clamp_min(EPS)
            # Higher accumulated NLL indicates a harder sample and a lower target confidence.
            target_gap = (history.roll(-1, 0) - history) / span
            quality = result['quality']
            gap = quality - quality.roll(-1, 0)
            valid = mask & mask.roll(-1, 0) & (target_gap != 0)
            loss = loss + masked_mean((target_gap.abs() - target_gap.sign() * gap).relu(), valid)
        return loss
