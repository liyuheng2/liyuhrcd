"""HRD-CF-M-HCRD-TD-RIB v3-S and matched controls."""
from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F

from model import EPS, entropy, jsd


MODES = {
    'v3s_fixed': 'fixed', 'v3s_full': 'dynamic', 'stochastic': 'stochastic',
    'deterministic': 'deterministic', 'shuffled': 'shuffled', 'reverse': 'reverse',
    'old_risk': 'fixed', 'without_o': 'dynamic', 'without_a': 'dynamic',
    'without_c': 'dynamic', 'without_u': 'dynamic', 'utility_only': 'dynamic',
    'without_hierarchy': 'dynamic', 'with_swap_task': 'dynamic',
}
FUSION_FRACTION = 0.25


def average_probabilities(p, mask):
    return (p * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)


def fusion_weights(utility, mask, rule='conservative'):
    uniform = mask / mask.sum(1, keepdim=True)
    weighted = utility.masked_fill(~mask, -torch.inf).softmax(1)
    if rule == 'uniform':
        return uniform
    if rule == 'softmax':
        return weighted
    return (1 - FUSION_FRACTION) * uniform + FUSION_FRACTION * weighted


class Classifier(nn.Module):
    def __init__(self, dims, classes, width):
        super().__init__()
        self.adapters = nn.ModuleList(nn.Linear(dim, width) for dim in dims)
        self.identity = nn.Embedding(len(dims), width)
        self.shared = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.semantic = nn.Linear(width, width)
        self.classifier = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, classes))

    def encode(self, inputs):
        mapped = torch.stack([adapter(x) for adapter, x in zip(self.adapters, inputs)], dim=1)
        hidden = self.shared(mapped + self.identity.weight)
        z = self.semantic(hidden)
        return hidden, z, self.classifier(z).softmax(-1)

    def forward(self, inputs, mask):
        _, _, p = self.encode(inputs)
        return {'single': p, 'p': average_probabilities(p, mask)}


class V3S(nn.Module):
    def __init__(self, classifier, modalities, classes, method, width, latent):
        super().__init__()
        self.backbone = classifier
        self.method = method
        self.ib = MODES[method]
        self.stochastic = self.ib != 'deterministic'
        self.hierarchy = method != 'without_hierarchy'
        self.active = tuple(method != 'without_' + task and
                            (method != 'utility_only' or task == 'u') for task in 'oacu')
        self.active += (method == 'with_swap_task',)
        self.tasks = 5 if self.active[4] else 4
        base = width * 2 + classes + 2
        self.sizes = (base, base + 1, width * 2 + 6, base + classes + 3 + modalities,
                      width * 2 + 6)[:self.tasks]
        self.value_size = max(self.sizes)
        self.task_identity = nn.Embedding(self.tasks, width)
        self.modality_identity = nn.Embedding(modalities, width)
        self.risk_encoder = nn.Sequential(nn.Linear(self.value_size * 2 + width * 2, width), nn.GELU())
        self.mean = nn.Linear(width, latent)
        self.log_variance = nn.Linear(width, latent)
        self.risk_head = nn.Linear(latent + width * 2, 1)
        left, right = zip(*combinations(range(modalities), 2))
        self.register_buffer('left', torch.tensor(left))
        self.register_buffer('right', torch.tensor(right))
        self.register_buffer('incidence', F.one_hot(self.left, modalities).float()
                             + F.one_hot(self.right, modalities).float())
        self.register_buffer('feature_masks', torch.arange(self.value_size)[None, :]
                             < torch.tensor(self.sizes)[:, None])

    def predict(self, value, identity, task):
        context = torch.cat((identity, self.task_identity.weight[task].expand_as(identity)), -1)
        context = context.expand(*value.shape[:-1], context.shape[-1])
        if task == 2 and self.active[4]:
            # C and the optional swap control have identical input formats.
            return self.predict_pair_tasks(value, identity)
        return self.posterior(value, context, self.feature_masks[task])

    def predict_pair_tasks(self, value, identity):
        contexts = torch.stack([torch.cat((identity, self.task_identity.weight[t].expand_as(identity)), -1)
                                for t in (2, 4)])
        contexts = contexts[:, None].expand(-1, value.shape[0], -1, -1)
        return self.posterior(value.unsqueeze(0).expand(2, -1, -1, -1), contexts, self.feature_masks[2])

    def posterior(self, value, context, feature_mask):
        padded = F.pad(value, (0, self.value_size - value.shape[-1]))
        h = self.risk_encoder(torch.cat((padded, feature_mask.expand_as(padded), context), -1))
        mean = self.mean(h)
        kl = mean.new_zeros(mean.shape[:-1])
        z = mean
        if self.training and self.stochastic:
            logvar = self.log_variance(h).clamp(-8, 5)
            z = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
            if self.ib != 'stochastic':
                kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).mean(-1)
        logits = self.risk_head(torch.cat((z, context), -1)).squeeze(-1)
        return logits.sigmoid(), kl, logits

    def forward(self, inputs, mask):
        h, z, p = self.backbone.encode(inputs)
        h, z, pr = h.detach(), z.detach(), p.detach()
        count = mask.sum(1, keepdim=True)
        prob_sum = (pr * mask[..., None]).sum(1, keepdim=True)
        others = (prob_sum - pr * mask[..., None]) / (count - 1).clamp_min(1)[..., None]
        others = others * (count > 1)[..., None]
        e = entropy(pr)
        value = torch.cat((h, z, pr, e[..., None], pr.amax(-1, keepdim=True)), -1)
        left, right = self.left, self.right
        pair_mask = mask[:, left] & mask[:, right]
        empty = p.new_zeros(mask.shape)
        empty_pair = p.new_zeros(pair_mask.shape)
        o, kl_o, log_o = self.predict(value, self.modality_identity.weight, 0) if self.active[0] else (empty,) * 3
        cond_o = o.detach() if self.hierarchy else empty
        a_input = torch.cat((value, cond_o[..., None]), -1)
        a, kl_a, log_a = self.predict(a_input, self.modality_identity.weight, 1) if self.active[1] else (empty,) * 3
        cond_a = a.detach() if self.hierarchy else empty
        c, kl_c, log_c, swap, kl_swap, log_swap = (empty_pair,) * 6
        if self.active[2]:
            stats = [(x[:, left] + x[:, right]) / 2 for x in (cond_o, cond_a)]
            stats += [(x[:, left] - x[:, right]).abs() for x in (cond_o, cond_a, e)]
            stats += [jsd(pr[:, left], pr[:, right])]
            pair = torch.cat((torch.stack(stats, -1), (z[:, left] - z[:, right]).abs(),
                              z[:, left] + z[:, right]), -1)
            pair_id = (self.modality_identity(left) + self.modality_identity(right)) / 2
            prediction, kl, logits = self.predict(pair, pair_id, 2)
            if self.active[4]:
                c, swap = prediction.unbind(0)
                kl_c, kl_swap = kl.unbind(0)
                log_c, log_swap = logits.unbind(0)
            else:
                c, kl_c, log_c = prediction, kl, logits
        conflict = ((c.detach() * pair_mask) @ self.incidence) / (count - 1).clamp_min(1)
        u_input = torch.cat((value, others, torch.stack((cond_o, cond_a,
                            conflict if self.hierarchy else empty), -1),
                            mask[:, None].expand(-1, mask.shape[1], -1)), -1)
        u, kl_u, log_u = (self.predict(u_input, self.modality_identity.weight, 3) if self.active[3]
                         else (empty + 0.5, empty, empty))
        utility = 2 * u - 1
        alpha = fusion_weights(utility, mask)
        p_fused = (p * alpha.detach()[..., None]).sum(1)
        n_pairs = pair_mask.sum(1)
        aggregate = torch.stack(((o * mask).sum(1) / count.squeeze(1),
                                 (a * mask).sum(1) / count.squeeze(1),
                                 ((c * pair_mask).sum(1) / n_pairs.clamp_min(1)).masked_fill(n_pairs == 0, torch.nan)), 1)
        return dict(p=p_fused, single=p, alpha=alpha, score=p_fused.amax(-1),
                    risks=(o, a, c, u, swap), kl=(kl_o, kl_a, kl_c, kl_u, kl_swap),
                    risk_logits=(log_o, log_a, log_c, log_u, log_swap),
                    utility=utility, mod_conflict=conflict, risk_vector=aggregate)


def leave_one_out(true_p, mask):
    count = mask.sum(1, keepdim=True)
    total = (true_p * mask).sum(1, keepdim=True)
    all_p = total / count
    without = (total - true_p * mask) / (count - 1).clamp_min(1)
    return (all_p.clamp_min(EPS).log() - without.clamp_min(EPS).log()).tanh()


def task_targets(source_p, current_p, reference_p, source_y, y, mask, swapped, left, right):
    source = source_p.gather(-1, source_y[..., None]).squeeze(-1).clamp_min(EPS)
    observed = current_p.gather(-1, source_y[..., None]).squeeze(-1).clamp_min(EPS)
    target = y[:, None, None].expand(-1, mask.shape[1], 1)
    current = current_p.gather(-1, target).squeeze(-1).clamp_min(EPS)
    reference = reference_p.gather(-1, target).squeeze(-1).clamp_min(EPS)
    touched = swapped[:, left] | swapped[:, right]
    conflict = 1 - ((current[:, left] + current[:, right])
                    / (reference[:, left] + reference[:, right])).clamp_max(1)
    utility = leave_one_out(current, mask)
    return (1 - (observed / source).clamp_max(1), 1 - source,
            conflict * touched, (1 + utility) / 2, touched.float())


def proxy_targets(source, current, strength, targets, left, right):
    h0, h = entropy(source), entropy(current)
    proxy_c = (1 - h0[:, left]) * (1 - h0[:, right]) * jsd(source[:, left], source[:, right])
    return strength, (1 + h - h0) / 2, proxy_c, *targets[3:]


def task_masks(mask, left, right):
    pairs = mask[:, left] & mask[:, right]
    return mask, mask, pairs, mask & (mask.sum(1, keepdim=True) > 1), pairs

