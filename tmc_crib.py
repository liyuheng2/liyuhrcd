"""TMC-HCRD-CRIB v2 with a shared conditional Gaussian bottleneck."""
from math import log

import torch
from torch import nn
from torch.nn import functional as F

from baselines import evidential_loss
from tmc_hcrd import EPS, TMCHCRD, entropy, fuse, js, mean_valid


VARIANTS = ('full', 'context_only', 'no_ib', 'fixed_ib', 'no_responsibility', 'no_harm', 'linear')


def raw_predictions(evidence, mask):
    """DS fusion for all available modalities and for each leave-one-out mask."""
    masked = evidence * mask[..., None]
    all_alpha = fuse(masked)
    # Each factor is >= 1. Divide out one factor to form each counterfactual.
    k = evidence.shape[-1]
    without = 1 + (all_alpha[:, None] - 1 - masked) / (1 + masked / k)
    return all_alpha / all_alpha.sum(-1, keepdim=True), without / without.sum(-1, keepdim=True)


def discount_evidence(evidence, reliability, mask, linear=False):
    reliability = reliability.to(evidence.dtype)
    scaled = evidence * (reliability * mask)[..., None]
    if linear:
        return scaled
    return scaled / (1 + (1 - reliability[..., None]) * evidence.mean(-1, keepdim=True))


class CRIB(nn.Module):
    def __init__(self, width, context, latent, variant):
        super().__init__()
        self.width, self.context, self.variant = width, context, variant
        self.task = nn.Embedding(3, 4)
        self.encoder = nn.Sequential(nn.Linear(width + context + 5, 64), nn.GELU(), nn.Linear(64, 2 * latent))
        self.head = nn.Linear(latent + context + 4, 1)

    def forward(self, features, context, task, intervention=None):
        context = F.pad(context.detach(), (0, self.context - context.shape[-1]))
        features = F.pad(features, (0, self.width - features.shape[-1]))
        conditions = torch.cat((context, self.task.weight[task].expand(*context.shape[:-1], 4)), -1)
        flag = torch.ones_like(features[..., :1])
        if self.variant == 'context_only':
            flag, features = torch.zeros_like(flag), torch.zeros_like(features)
        mu, logvar = self.encoder(torch.cat((features, conditions, flag), -1)).chunk(2, -1)
        logvar = logvar.clamp(-8, 6)
        stochastic = self.variant not in {'context_only', 'no_ib'}
        kl = torch.zeros_like(mu[..., 0])
        if stochastic:
            if self.variant == 'fixed_ib':
                prior_mu, prior_logvar = 0, torch.zeros_like(logvar)
            else:
                prior_mu, prior_logvar = self.encoder(torch.cat((torch.zeros_like(features), conditions,
                                                                 torch.zeros_like(flag)), -1)).chunk(2, -1)
                prior_logvar = prior_logvar.clamp(-8, 6)
            kl = 0.5 * (prior_logvar - logvar + (logvar - prior_logvar).exp()
                        + (mu - prior_mu).square() * (-prior_logvar).exp() - 1).sum(-1)
        z = mu + (logvar / 2).exp() * torch.randn_like(mu) if self.training and stochastic else mu
        if intervention == 'zero':
            z = torch.zeros_like(z)
        elif intervention == 'shuffle':
            z = z.roll(1, 0)
        return self.head(torch.cat((z, conditions), -1)).squeeze(-1), kl


class TMCcrib(TMCHCRD):
    def __init__(self, dims, classes, variant, width=128, latent=16):
        super().__init__(dims, classes, 'tmc', width, latent)
        self.variant = variant
        m = len(dims)
        self.register_buffer('modality', torch.eye(m))
        self.observation = nn.Linear(width, 1)
        self.crib = CRIB(3 * width, max(12 + 2 * m, 6 + 3 * classes + 2 * m), latent, variant)

    def forward(self, inputs, mask, intervention=None):
        h, e = self.evidence(inputs)
        p, h_entropy, u = self.distribution(e)
        log_s = (1 + e).sum(-1).log()
        o_logit = self.observation(h).squeeze(-1)
        o = o_logit.sigmoid()
        identity = self.modality.expand(len(mask), -1, -1)
        a_logit, kl_a = self.crib(h, torch.cat((torch.stack((o, h_entropy, u, log_s), -1), identity), -1),
                                 0, intervention)
        a = a_logit.sigmoid()

        i, j = self.i, self.j
        pair = mask[:, i] & mask[:, j]
        incidence = self.pair_i + self.pair_j
        remaining = (1 - incidence) * (mask * (1 - o))[:, None, :]
        support = remaining.sum(-1)
        reference = torch.where(support[..., None] > EPS, (remaining @ p) / support[..., None].clamp_min(EPS),
                                1 / self.classes)
        raw = js(p[:, i], p[:, j])
        ref_i, ref_j = js(p[:, i], reference), js(p[:, j], reference)
        source, target = torch.cat((i, j)), torch.cat((j, i))
        hs, ht = h[:, source], h[:, target]
        v_c = torch.cat((hs, ht, (hs - ht).abs()), -1)
        scalars = [value[:, endpoint] for value in (o, a, h_entropy, u) for endpoint in (source, target)]
        scalars.extend((raw.repeat(1, 2), torch.cat((ref_i, ref_j), 1), torch.cat((ref_j, ref_i), 1),
                        (support > EPS).repeat(1, 2)))
        s_c = torch.cat((torch.stack(scalars, -1), identity[:, source], identity[:, target]), -1)
        directed, kl_c = self.crib(v_c, s_c, 1, intervention)
        left, right = directed.chunk(2, 1)
        conflict_logits = torch.stack((torch.zeros_like(left), left, right), -1)
        pi = conflict_logits.softmax(-1)
        c = ((pi[..., 1] * pair) @ self.pair_i + (pi[..., 2] * pair) @ self.pair_j)
        c = c / (pair.float() @ incidence).clamp_min(1)
        conflict = 1 - pi[..., 0]

        all_p, without_p = raw_predictions(e, mask)
        s_d = torch.cat((torch.stack((o, a, c, h_entropy, u, log_s), -1), p,
                         all_p[:, None].expand_as(p), without_p, identity, mask[:, None].expand_as(identity)), -1)
        d_logit, kl_d = self.crib(h, s_d, 2, intervention)
        delta = 2 * d_logit.sigmoid() - 1
        d = 1 - delta.relu()
        if self.variant == 'no_harm':
            d = (1 - o) * (1 - a) / (1 + c)
        d = torch.where(mask.sum(1, keepdim=True) > 1, d, 1)
        alpha = fuse(discount_evidence(e, d, mask, self.variant == 'linear'))
        strength = alpha.sum(-1)
        weights = mask * (1 - o)
        obs_support = weights.sum(1) > EPS
        return dict(p=alpha / strength[:, None], alpha=alpha, score=1 - self.classes / strength,
                    single_alpha=1 + e, o=o, a=a, c=c, o_logit=o_logit, a_logit=a_logit, d_logit=d_logit,
                    conflict=conflict, conflict_logits=conflict_logits, raw=raw, pair_mask=pair, delta=delta,
                    discount=d * mask, kl_a=kl_a, kl_c=sum(kl_c.chunk(2, 1)), kl_d=kl_d,
                    risks=torch.stack(((o * mask).sum(1) / mask.sum(1).clamp_min(1),
                                      torch.where(obs_support, (a * weights).sum(1) / weights.sum(1).clamp_min(EPS),
                                                  (a * mask).sum(1) / mask.sum(1).clamp_min(1)),
                                      (conflict * pair).sum(1) / pair.sum(1).clamp_min(1)), -1),
                    ambiguity_supported=obs_support, conflict_available=pair.any(1))


def classification(result, y, mask, epoch, swapped=None):
    anneal = min(1., (epoch + 1) / 10)
    fused = evidential_loss(result['alpha'], y, anneal).mean()
    valid = mask if swapped is None else mask & ~swapped
    single = evidential_loss(result['single_alpha'].flatten(0, 1), y[:, None].expand_as(mask).flatten(), anneal)
    return fused + mean_valid(single.view_as(mask), valid)


@torch.no_grad()
def teacher_targets(teacher, clean, current, mask, y):
    _, ambiguity, _ = teacher.distribution(teacher.evidence(clean)[1])
    all_p, without_p = raw_predictions(teacher.evidence(current)[1], mask)
    onehot = F.one_hot(y, teacher.classes).float()
    delta = ((all_p - onehot).square().sum(-1, keepdim=True)
             - (without_p - onehot[:, None]).square().sum(-1)) / 2
    return ambiguity, delta


def risk_losses(model, output, mask, polluted, cross_swaps, targets):
    ambiguity, delta = targets
    pair = output['pair_mask']
    multiple = mask & (mask.sum(1, keepdim=True) > 1)
    labels = cross_swaps[:, model.i].long() + 2 * cross_swaps[:, model.j].long()
    bce = F.binary_cross_entropy_with_logits
    loss = mean_valid(bce(output['o_logit'], polluted.float(), reduction='none'), mask)
    loss += mean_valid(bce(output['a_logit'], ambiguity, reduction='none'), mask)
    loss += mean_valid(bce(output['d_logit'], (1 + delta) / 2, reduction='none'), multiple)
    count = 2 + multiple.any().int()
    pair_ce = F.cross_entropy(output['conflict_logits'].flatten(0, 1), labels.flatten(), reduction='none')
    if model.variant != 'no_responsibility':
        loss += mean_valid(pair_ce.view_as(pair), pair)
        count += pair.any().int()
    kl = (mean_valid(output['kl_a'], mask) + mean_valid(output['kl_c'], pair)
          + mean_valid(output['kl_d'], multiple)) / (1 + pair.any().int() + multiple.any().int())
    return loss / count, kl
