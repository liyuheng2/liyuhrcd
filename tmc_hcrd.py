"""TMC evidence fusion with cause-aware discounts and a conditional risk bottleneck."""
from itertools import combinations
from math import log

import torch
from torch import nn
from torch.nn import functional as F

from baselines import Baseline, evidential_loss


METHODS = ('tmc', 'simple', 'hcrd', 'fixed_ib', 'dynamic_ib', 'full',
           'no_o', 'no_a', 'no_c', 'only_o', 'only_a')
EPS = 1e-8


def entropy(p):
    return -(p * p.clamp_min(EPS).log()).sum(-1)


def js(p, q):
    return ((entropy((p + q) / 2) - (entropy(p) + entropy(q)) / 2) / log(2)).clamp(0, 1)


def fuse(evidence):
    # Cancelling the DS normalization in b/u yields this equivalent update.
    # A missing view contributes zero evidence, the neutral opinion.
    total = torch.zeros_like(evidence[:, 0])
    for view in evidence.unbind(1):
        total = total + view + total * view / evidence.shape[-1]
    return 1 + total


def mean_valid(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


class RiskBottleneck(nn.Module):
    def __init__(self, variational, latent):
        super().__init__()
        self.variational = variational
        self.task = nn.Embedding(2, 4)
        self.encoder = nn.Sequential(nn.Linear(11, 64), nn.GELU(),
                                     nn.Linear(64, latent * (2 if variational else 1)))
        self.head = nn.Linear(latent + 4, 1)

    def forward(self, inputs, task, sample):
        embedding = self.task.weight[task].expand(*inputs.shape[:-1], -1)
        latent = self.encoder(torch.cat((inputs, embedding), dim=-1))
        kl = inputs.new_zeros(inputs.shape[:-1])
        if self.variational:
            mean, logvar = latent.chunk(2, dim=-1)
            logvar = logvar.clamp(-8, 6)
            kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).mean(-1)
            latent = mean + (0.5 * logvar).exp() * torch.randn_like(mean) if sample else mean
        output = self.head(torch.cat((latent, embedding), dim=-1)).squeeze(-1)
        return (output.tanh() if task == 0 else output.sigmoid()), kl


class TMCHCRD(Baseline):
    def __init__(self, dims, classes, method, width=128, latent=16):
        super().__init__(dims, classes, 'tmc', 0, width)
        self.method = method
        self.classes = classes
        pairs = list(combinations(range(len(dims)), 2))
        self.register_buffer('i', torch.tensor([i for i, _ in pairs]))
        self.register_buffer('j', torch.tensor([j for _, j in pairs]))
        self.register_buffer('pair_i', F.one_hot(self.i, len(dims)).float())
        self.register_buffer('pair_j', F.one_hot(self.j, len(dims)).float())
        self.register_buffer('other', 1 - torch.eye(len(dims)))
        if self.uses_risk:
            self.observation = nn.Linear(width, 1)
            self.rib = RiskBottleneck(method != 'hcrd', latent)

    @property
    def uses_risk(self):
        return self.method not in {'tmc', 'simple'}

    def evidence(self, inputs):
        features = torch.stack([layer(x) for layer, x in zip(self.encoders, inputs)], dim=1)
        evidence = torch.stack([head(h) for head, h in zip(self.heads, features.unbind(1))], dim=1)
        return features, F.softplus(evidence)

    def distribution(self, evidence):
        alpha = 1 + evidence
        strength = alpha.sum(-1)
        p = alpha / strength[..., None]
        return p, entropy(p) / log(self.classes), self.classes / strength

    def forward(self, inputs, mask, warmup=False, risk_only=False):
        features, evidence = self.evidence(inputs)
        p, h, u = self.distribution(evidence)
        discount = mask.float()
        result = {'single_alpha': 1 + evidence, 'u': u}
        if not warmup and self.method == 'simple':
            discount = discount * (1 - u)
        if not warmup and self.uses_risk:
            if risk_only:
                features, evidence, p, h, u = (t.detach() for t in (features, evidence, p, h, u))
            i, j = self.i, self.j
            pair_mask = mask[:, i] & mask[:, j]
            o_logit = self.observation(features).squeeze(-1)
            o = o_logit.sigmoid()
            raw = js(p[:, i], p[:, j])
            a_input = torch.stack((o, evidence.sum(-1).log1p(), h, u), dim=-1)
            delta, kl_a = self.rib(F.pad(a_input, (0, 3)), 0, self.training)
            # Sort the members symmetrically via their mean and absolute difference.
            c_input = [raw]
            for stat in (o, h, u):
                c_input.extend(((stat[:, i] + stat[:, j]) / 2, (stat[:, i] - stat[:, j]).abs()))
            normal, kl_c = self.rib(torch.stack(c_input, dim=-1), 1, self.training)
            a = (h - delta).clamp(0, 1)
            excess = (raw - normal).clamp(0, 1) * pair_mask
            r_o = o if self.method not in {'no_o', 'only_a'} else torch.zeros_like(o)
            r_a = a if self.method not in {'no_a', 'only_o'} else torch.zeros_like(a)
            trust = (1 - r_o) * (1 - r_a) * mask
            trust_sum = trust @ self.other
            reference = (p * trust[..., None]).transpose(1, 2) @ self.other
            reference = reference.transpose(1, 2) / trust_sum[..., None].clamp_min(EPS)
            q = js(p, reference) * (trust_sum > EPS)
            share = (q[:, i] + EPS) / (q[:, i] + q[:, j] + 2 * EPS)
            responsibility = ((excess * share * trust[:, j]) @ self.pair_i
                              + (excess * (1 - share) * trust[:, i]) @ self.pair_j)
            responsibility = responsibility / (trust_sum + EPS)
            if self.method in {'full', 'no_o', 'no_a'}:
                discount = trust / (1 + responsibility)
            else:
                discount = trust
            result.update(o=o, o_logit=o_logit, delta=delta, normal=normal, a=a, excess=excess,
                          raw=raw, responsibility=responsibility * mask, pair_mask=pair_mask,
                          kl_a=kl_a, kl_c=kl_c)
        alpha = fuse(evidence * discount[..., None])
        strength = alpha.sum(-1)
        return dict(result, alpha=alpha, p=alpha / strength[:, None],
                    score=1 - self.classes / strength, discount=discount)

    @torch.no_grad()
    def references(self, teacher, clean, current):
        p_clean, h_clean, _ = teacher.distribution(teacher.evidence(clean)[1])
        _, h_current, _ = teacher.distribution(teacher.evidence(current)[1])
        return h_current - h_clean, js(p_clean[:, self.i], p_clean[:, self.j]), h_clean

    def risk_loss(self, result, mask, strength, refs, beta):
        delta, normal, entropy_clean = refs
        pair = result['pair_mask']
        o = F.binary_cross_entropy_with_logits(result['o_logit'], strength, reduction='none')
        a = (result['delta'] - delta).square()
        c = (result['normal'] - normal).square()
        risk = (mean_valid(o, mask) + mean_valid(a, mask) + mean_valid(c, pair)) / 3
        wa = wc = 1
        if self.method not in {'fixed_ib', 'hcrd'}:
            with torch.no_grad():
                c_proxy = (result['raw'] - normal).clamp(0, 1) * pair
                count = pair.float() @ (self.pair_i + self.pair_j)
                c_modal = c_proxy @ (self.pair_i + self.pair_j) / count.clamp_min(1)
                support = strength * (1 - entropy_clean)
                wa = ((1 + support) / (1 + c_modal)).clamp(0.5, 2)
                wc = ((1 + (support[:, self.i] + support[:, self.j]) / 2)
                      / (1 + c_proxy)).clamp(0.5, 2)
        kl = (mean_valid(wa * result['kl_a'], mask) + mean_valid(wc * result['kl_c'], pair)) / 2
        return risk + beta * kl


def task_loss(result, y, mask, epoch):
    anneal = min(1.0, (epoch + 1) / 10)
    fused = evidential_loss(result['alpha'], y, anneal).mean()
    alpha = result['single_alpha'].flatten(0, 1)
    labels = y[:, None].expand_as(mask).reshape(-1)
    return fused + mean_valid(evidential_loss(alpha, labels, anneal).view_as(mask), mask)
