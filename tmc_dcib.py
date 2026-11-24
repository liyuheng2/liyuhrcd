"""Frozen TMC diagnostics, decision bottleneck and analytic DS adjustment."""
import torch
from torch import nn
from torch.nn import functional as F

from tmc_hcrd import EPS, fuse, js, mean_valid


VARIANTS = ('dcib', 'context_only', 'no_ib', 'category_ib', 'direction_ib',
            'equal_direction_ib', 'fixed_ib', 'no_hcrd', 'no_observation',
            'no_ambiguity', 'no_conflict')


class HCRDLite(nn.Module):
    def __init__(self, backbone, width):
        super().__init__()
        m = len(backbone.encoders)
        self.register_buffer('identity', torch.eye(m))
        self.register_buffer('incidence', backbone.pair_i + backbone.pair_j)
        self.register_buffer('i', backbone.i.clone())
        self.register_buffer('j', backbone.j.clone())
        self.observation = nn.Linear(width, 1)
        self.responsibility = nn.Linear(5 * backbone.classes + 10 + 2 * m, 1)

    def forward(self, h, p, a, u, mask):
        i, j = self.i, self.j
        pair = mask[:, i] & mask[:, j]
        obs_logits = self.observation(h).squeeze(-1)
        o = obs_logits.sigmoid()
        remaining = (1 - self.incidence) * (mask * (1 - o))[:, None]
        support = remaining.sum(-1)
        reference = torch.where(support[..., None] > EPS,
                                (remaining @ p) / support.clamp_min(EPS)[..., None], 1 / p.shape[-1])
        raw = js(p[:, i], p[:, j])

        def ordered(left, right):
            pl, pr = p[:, left], p[:, right]
            diff = pl - pr
            scalars = torch.stack((raw, o[:, left], o[:, right], a[:, left], a[:, right],
                                   u[:, left], u[:, right], js(pl, reference), js(pr, reference),
                                   support > EPS), -1)
            tags = torch.cat((self.identity[left], self.identity[right]), -1)
            return torch.cat((pl, pr, pl * pr, diff, diff.abs(), scalars,
                              tags.expand(len(mask), -1, -1)), -1)

        left = self.responsibility(ordered(i, j)).squeeze(-1)
        right = self.responsibility(ordered(j, i)).squeeze(-1)
        logits = torch.stack((torch.zeros_like(left), left, right), -1)
        pi = logits.softmax(-1) * pair[..., None]
        c = (pi[..., 1] @ self.identity[i] + pi[..., 2] @ self.identity[j])
        c = c / (pair.float() @ self.incidence).clamp_min(1)
        conflict = pi[..., 1:].sum(-1)
        risks = torch.stack(((o * mask).sum(1) / mask.sum(1).clamp_min(1),
                             (a * mask).sum(1) / mask.sum(1).clamp_min(1),
                             conflict.sum(1) / pair.sum(1).clamp_min(1)), -1)
        return dict(obs_logits=obs_logits, o=o * mask, a=a * mask, c=c, conflict=conflict,
                    conflict_logits=logits, pair=pair, risks=risks)

    def loss(self, output, mask, strength, cross):
        labels = cross[:, self.i].long() + 2 * cross[:, self.j].long()
        obs = F.binary_cross_entropy_with_logits(output['obs_logits'], strength, reduction='none')
        conflict = F.cross_entropy(output['conflict_logits'].transpose(1, 2), labels, reduction='none')
        pair = output['pair']
        return (mean_valid(obs, mask) + mean_valid(conflict, pair)) / (1 + pair.any().int())


def geometry(evidence, mask):
    e = evidence * mask[..., None]
    k = e.shape[-1]
    alpha = fuse(e)
    # The DS evidence contribution of each view factors out of the joint opinion.
    other = ((alpha[:, None] - 1 - e) / (1 + e / k)).clamp_min(0)
    p0 = alpha / alpha.sum(-1, keepdim=True)
    other_strength = (other + 1).sum(-1)
    without = (other + 1) / other_strength[..., None]
    direction = without - p0[:, None]
    lo, hi = direction.amin(-1), direction.amax(-1)
    span = hi - lo
    norm = direction.square().sum(-1)
    valid = mask & (mask.sum(1, keepdim=True) > 1) & (span > EPS) & (norm > EPS ** 2)
    return dict(e=e, mask=mask, alpha=alpha, p0=p0, without=without, direction=direction, lo=lo,
                span=span, norm=norm, valid=valid, other_strength=other_strength,
                contribution=(e * (1 + other / k)).sum(-1))


def context(h, p, u, g, risks, variant):
    mask = g['mask']
    o, a, conflict, c = (risks[k] for k in ('o', 'a', 'conflict', 'c'))
    if variant in {'no_hcrd', 'no_observation'}:
        o = torch.zeros_like(o)
    if variant in {'no_hcrd', 'no_ambiguity'}:
        a = torch.zeros_like(a)
    if variant in {'no_hcrd', 'no_conflict'}:
        conflict, c = torch.zeros_like(conflict), torch.zeros_like(c)
    # Preserve the ordinary predictive context when removing risk inputs.
    s = torch.cat((mask, o, a, conflict, c, (p * mask[..., None]).flatten(1), u * mask,
                   g['p0'], (g['without'] * mask[..., None]).flatten(1)), -1)
    return (h * mask[..., None]).flatten(1), s


class DCIB(nn.Module):
    def __init__(self, feature_dim, context_dim, classes, width, latent, variant):
        super().__init__()
        self.variant = variant
        self.encoder = nn.Sequential(nn.Linear(feature_dim + context_dim + 1, width), nn.GELU(),
                                     nn.Linear(width, 2 * latent))
        self.decoder = nn.Linear(latent + context_dim, classes)

    def encode(self, v, s, flag):
        mu, logvar = self.encoder(torch.cat((v, s, flag), -1)).chunk(2, -1)
        return mu, logvar.clamp(-8, 6)

    def forward(self, v, s, p0, noise=None, latent_control=None):
        flag = torch.ones_like(v[:, :1])
        if self.variant == 'context_only':
            v, flag = torch.zeros_like(v), torch.zeros_like(flag)
        mu, logvar = self.encode(v, s, flag)
        stochastic = self.variant not in {'context_only', 'no_ib'}
        kl = mu.new_zeros(len(mu))
        if stochastic:
            if self.variant == 'fixed_ib':
                prior_mu, prior_logvar = torch.zeros_like(mu), torch.zeros_like(logvar)
            else:
                prior_mu, prior_logvar = self.encode(torch.zeros_like(v), s, torch.zeros_like(flag))
            kl = .5 * (prior_logvar - logvar + (logvar - prior_logvar).exp()
                       + (mu - prior_mu).square() * (-prior_logvar).exp() - 1).sum(-1)
        z = mu.unsqueeze(0)
        if stochastic and noise is not None:
            z = z + (.5 * logvar).exp() * noise
        if latent_control == 'zero':
            z = torch.zeros_like(z)
        elif latent_control == 'shuffle':
            z = z.roll(1, dims=1)
        residual = self.decoder(torch.cat((z, s.expand(z.shape[0], -1, -1)), -1))
        q = (p0.clamp_min(EPS).log() + residual).softmax(-1).mean(0)
        return q, kl


def objective(model, q, kl, g, y, lam, beta):
    label = F.one_hot(y, q.shape[-1]).to(q.dtype)
    projected = torch.stack((q, label))[:, :, None] * g['direction']
    targets = ((projected.sum(-1) - g['lo']) / g['span'].clamp_min(EPS)).clamp(0, 1)
    pred, actual = targets.unbind(0)
    valid = g['valid']
    weights = valid.float() if model.variant == 'equal_direction_ib' else valid * g['span'].square()
    weights = weights / weights.sum(1, keepdim=True).clamp_min(EPS ** 2)
    bce = F.binary_cross_entropy(pred.clamp(1e-7, 1 - 1e-7), actual, reduction='none')
    direction = (weights * bce).sum(1).mean()
    category = F.nll_loss(q.clamp_min(EPS).log(), y)
    lam = {'category_ib': 0., 'direction_ib': 1.}.get(model.variant, lam)
    return (1 - lam) * category + lam * direction + beta * kl.mean()


def proposal(q, g, discrete=False):
    first = ((q - g['p0'])[:, None] * g['direction']).sum(-1)
    if discrete:
        w = (first - .5 * g['norm'] > 0).to(q.dtype)
    else:
        w = (first / g['norm'].clamp_min(EPS ** 2)).clamp(0, 1)
    gain = (w * first - .5 * w.square() * g['norm']).masked_fill(~g['valid'], 0)
    best, index = gain.max(1)
    return best, index, w.gather(1, index[:, None]).squeeze(1), first


def adjust(g, action, threshold=0.):
    gain, index, w, _ = action
    selected = F.one_hot(index, g['mask'].shape[1]).bool() & (gain > threshold)[:, None]
    w = torch.where(selected, w[:, None], 0.)
    strength, contribution = g['other_strength'], g['contribution']
    scale = strength * (1 - w) / (strength + w * contribution)
    alpha = fuse(g['e'] * scale[..., None])
    total = alpha.sum(-1)
    p = alpha / total[:, None]
    k, evidence = p.shape[-1], g['e'].sum(-1)
    reliability = scale * (k + evidence) / (k + scale * evidence)
    predictable = g['mask'].any(1)
    return dict(p=p, vacuity=k / total, discount=reliability * g['mask'], scale=scale,
                adjusted=selected.any(1), predictable=predictable,
                prediction=p.argmax(1).masked_fill(~predictable, -1))


@torch.inference_mode()
def predict(backbone, hcrd, dcib, inputs, mask, threshold=0., generator=None, samples=8):
    """Predict and return DS outputs and risks from standardized features and a Boolean mask.

    Use eval-mode modules and pass the calibrated threshold. None requests the TMC fallback.
    Pass a CPU generator to reproduce MC draws, or set samples=0 for posterior-mean inference.
    """
    h, evidence = backbone.evidence(inputs)
    p, ambiguity, u = backbone.distribution(evidence)
    risks = hcrd(h, p, ambiguity, u, mask)
    g = geometry(evidence, mask)
    v, s = context(h, p, u, g, risks, dcib.variant)
    noise = None
    if samples:
        latent = dcib.encoder[-1].out_features // 2
        noise = torch.randn((len(mask), samples, latent), generator=generator, dtype=v.dtype)
        noise = noise.transpose(0, 1).to(v.device)
    q, _ = dcib(v, s, g['p0'], noise)
    action = proposal(q, g)
    return dict(adjust(g, action, float('inf') if threshold is None else threshold),
                q=q, gain=action[0], risks=risks['risks'], conflict_available=risks['pair'].any(1))
