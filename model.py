"""HRD-CF-M-HCRD-RIB v2 and matched experimental ablations."""
from itertools import combinations
from math import log

import torch
from torch import nn
from torch.nn import functional as F


EPS = 1e-7
ABLATIONS = {
    'full', 'no_ib', 'fixed_ib', 'dynamic_kl', 'dynamic_risk', 'independent_ib',
    'shuffled_evidence', 'no_o', 'no_a', 'no_c', 'raw_jsd',
    'plain_jsd_target', 'no_csf', 'no_detach', 'no_unimodal',
}


def entropy(p):
    return -(p * p.clamp_min(EPS).log()).sum(-1) / log(p.shape[-1])


def jsd(left, right):
    middle = 0.5 * (left + right)
    log_middle = middle.clamp_min(EPS).log()
    left_kl = (left * (left.clamp_min(EPS).log() - log_middle)).sum(-1)
    right_kl = (right * (right.clamp_min(EPS).log() - log_middle)).sum(-1)
    return ((left_kl + right_kl) / (2 * log(2))).clamp(0, 1)


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def classification_loss(result, labels, mask, use_unimodal=True):
    fused = -result['p'].clamp_min(EPS).log().gather(1, labels[:, None]).mean()
    if not use_unimodal:
        return fused
    target = labels[:, None, None].expand(-1, mask.shape[1], 1)
    losses = -result['single'].clamp_min(EPS).log().gather(2, target).squeeze(2)
    return fused + ((mask * losses).sum(1) / mask.sum(1)).mean()


class RiskBottleneck(nn.Module):
    def __init__(self, width, latent):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(width * 2, width), nn.GELU())
        self.mean = nn.Linear(width, latent)
        self.log_variance = nn.Linear(width, latent)
        self.predict = nn.Linear(latent, 1)

    def forward(self, value, task, stochastic):
        hidden = self.encode(value)
        mean = self.mean(hidden)
        if stochastic:
            log_variance = self.log_variance(hidden).clamp(-8, 5)
            latent = mean + torch.randn_like(mean) * (0.5 * log_variance).exp()
            kl = 0.5 * (mean.square() + log_variance.exp() - 1 - log_variance).mean(-1)
        else:
            latent = mean
            kl = mean.new_zeros(mean.shape[:-1])
        prediction = self.predict(latent).squeeze(-1)
        return (prediction.tanh() if task == 1 else prediction.sigmoid()), kl


class HCRDRIB(nn.Module):
    def __init__(self, dims, classes, mode='full', width=128, latent=16):
        super().__init__()
        self.mode = mode
        self.adapters = nn.ModuleList(nn.Linear(dim, width) for dim in dims)
        self.identity = nn.Embedding(len(dims), width)
        self.shared = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.semantic = nn.Linear(width, width)
        self.observation = nn.Linear(width, width)
        self.classifier = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, classes))
        self.risk_input = nn.ModuleList(nn.Linear(width + extra, width) for extra in (2, 3, 5))
        self.task_identity = nn.Embedding(3, width)
        self.rib = nn.ModuleList(RiskBottleneck(width, latent) for _ in range(3 if mode == 'independent_ib' else 1))
        pairs = list(combinations(range(len(dims)), 2))
        self.register_buffer('left', torch.tensor([i for i, _ in pairs], dtype=torch.long))
        self.register_buffer('right', torch.tensor([j for _, j in pairs], dtype=torch.long))
        self.register_buffer('incidence_l', F.one_hot(self.left, len(dims)).float())
        self.register_buffer('incidence_r', F.one_hot(self.right, len(dims)).float())

    def encode(self, inputs):
        mapped = torch.stack([adapter(x) for adapter, x in zip(self.adapters, inputs)], dim=1)
        hidden = self.shared(mapped + self.identity.weight)
        semantic = self.semantic(hidden)
        return semantic, hidden, self.classifier(semantic).softmax(-1)

    def predict_risk(self, inputs, identity, task):
        features = self.risk_input[task](inputs)
        context = (identity + self.task_identity.weight[task]).expand_as(features)
        risk = self.rib[task if self.mode == 'independent_ib' else 0]
        return risk(torch.cat((features, context), dim=-1), task, self.training and self.mode != 'no_ib')

    def forward(self, inputs, mask, uniform=False):
        z, hidden, p = self.encode(inputs)
        uniform_weights = mask / mask.sum(1, keepdim=True)
        if uniform:
            return self.fuse(z, p, uniform_weights, p.new_ones(mask.shape[0]))
        detach = (lambda value: value) if self.mode == 'no_detach' else torch.Tensor.detach
        e, conf = entropy(p), p.max(-1).values
        d = jsd(p[:, self.left], p[:, self.right])
        o_features = self.observation(detach(hidden))
        inputs_o = torch.cat((o_features, detach(e)[..., None], detach(conf)[..., None]), -1)
        pred_o, kl_o = self.predict_risk(inputs_o, self.identity.weight, 0)
        r_o = torch.zeros_like(pred_o) if self.mode == 'no_o' else pred_o
        inputs_a = torch.cat((inputs_o, detach(r_o)[..., None]), -1)
        pred_a, kl_a = self.predict_risk(inputs_a, self.identity.weight, 1)
        r_a = e if self.mode == 'no_a' else (e - pred_a).clamp(0, 1)
        pair_features = [detach(d)]
        for risk in (detach(r_o), detach(r_a)):
            first, second = risk[:, self.left], risk[:, self.right]
            pair_features.extend(((first + second) / 2, (first - second).abs()))
        z_diff = (detach(z[:, self.left]) - detach(z[:, self.right])).abs()
        inputs_c = torch.cat((z_diff, torch.stack(pair_features, -1)), -1)
        pair_ids = (self.identity(self.left) + self.identity(self.right)) / 2
        pred_c, kl_c = self.predict_risk(inputs_c, pair_ids, 2)
        raw = d * (1 - r_a[:, self.left]) * (1 - r_a[:, self.right])
        r_c = (raw - pred_c).clamp_min(0)
        if self.mode == 'raw_jsd':
            r_c = d
        elif self.mode == 'no_c':
            r_c = torch.zeros_like(d)
        pair_mask = mask[:, self.left] & mask[:, self.right]
        t = (1 - r_o) * (1 - r_a)
        l, r = t[:, self.left], t[:, self.right]
        share = (1 - l + EPS) / (2 - l - r + 2 * EPS)
        weight_l, weight_r = pair_mask * r, pair_mask * l
        numerator = (weight_l * r_c * share) @ self.incidence_l + (weight_r * r_c * (1 - share)) @ self.incidence_r
        denominator = weight_l @ self.incidence_l + weight_r @ self.incidence_r
        responsibility = numerator / (denominator + EPS)
        reliability = t * (1 - responsibility)
        weights = mask * reliability
        mass = weights.sum(1, keepdim=True)
        alpha = torch.where(mass > EPS, weights / mass.clamp_min(EPS), uniform_weights)
        if self.mode == 'no_csf':
            alpha = uniform_weights
        score = (alpha * reliability).sum(1)
        result = self.fuse(z, p, detach(alpha), score)
        result.update(alpha=alpha, r_o=r_o, r_a=r_a, r_c=r_c, d=d, entropy=e,
                      pred_risk=(pred_o, pred_a, pred_c), kl=(kl_o, kl_a, kl_c),
                      masks=(mask, mask, pair_mask), responsibility=responsibility)
        return result

    def fuse(self, z, p, alpha, score):
        average = (alpha[..., None] * p).sum(1)
        joint = self.classifier((alpha[..., None] * z).sum(1)).softmax(-1)
        return {'p': (average + joint) / 2, 'single': p, 'score': score, 'alpha': alpha}

    @torch.no_grad()
    def references(self, clean, result, strength):
        _, _, clean_p = self.encode(clean)
        clean_h = entropy(clean_p)
        clean_d = jsd(clean_p[:, self.left], clean_p[:, self.right])
        confidence = (1 - clean_h[:, self.left]) * (1 - clean_h[:, self.right])
        delta_h = result['entropy'] - clean_h
        target_c = clean_d if self.mode == 'plain_jsd_target' else confidence * clean_d
        evidence_c = (confidence * (result['d'] - clean_d)).clamp_min(0)
        return (strength, delta_h, target_c), (strength, delta_h.abs(), evidence_c)

    def risk_losses(self, result, targets, evidence, rng):
        errors, rates = [], []
        names = ('no_o', 'no_a', 'no_c')
        zero = result['p'].new_zeros(())
        for task, (pred, target, delta, kl, mask) in enumerate(zip(
            result['pred_risk'], targets, evidence, result['kl'], result['masks']
        )):
            if self.mode == names[task] or (task == 2 and self.mode == 'raw_jsd') or not mask.any():
                continue
            if self.mode == 'shuffled_evidence':
                valid = delta[mask]
                delta = delta.clone()
                permutation = torch.randperm(len(valid), generator=rng).to(valid.device)
                delta[mask] = valid[permutation]
            weight = 1 + delta
            compress = weight if self.mode in {'full', 'dynamic_kl', 'independent_ib', 'shuffled_evidence',
                                                'no_o', 'no_a', 'no_c', 'raw_jsd', 'plain_jsd_target',
                                                'no_csf', 'no_detach', 'no_unimodal'} else 1
            retain = weight if self.mode not in {'fixed_ib', 'no_ib', 'dynamic_kl'} else 1
            error = ((pred - target) / (2 if task == 1 else 1)).square()
            rates.append(masked_mean(kl / compress, mask))
            errors.append(masked_mean(retain * error, mask))
        if not errors:
            return zero, zero
        return torch.stack(rates).mean(), torch.stack(errors).logsumexp(0) - log(len(errors))
