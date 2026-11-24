"""Shared conditional bottleneck and action-consistent DS fusion."""
import torch
from torch import nn
from torch.nn import functional as F

from baselines import evidential_loss
from tmc_hcrd import EPS, TMCHCRD, fuse, js, mean_valid


TASKS = ('A', 'C', 'D')
VARIANTS = ('acrib', 'no_ib', 'context_only', 'fixed_vib', 'rib', 'crib',
            'idab_crib', 'fixed_acrib', 'random_acrib', 'unbounded_acrib')


def probabilities(alpha):
    return alpha / alpha.sum(-1, keepdim=True)


def counterfactual(evidence, mask):
    """Compute raw, half-retained, and removed DS opinions without re-encoding."""
    evidence = evidence * mask[..., None]
    alpha = fuse(evidence)
    other = (alpha[:, None] - 1 - evidence).clamp_min(0) / (1 + evidence / evidence.shape[-1])
    half = evidence / 2
    candidates = 1 + torch.stack((other + half + other * half / evidence.shape[-1], other), dim=2)
    return alpha, candidates


def gains(alpha, candidates, y):
    target = F.one_hot(y, alpha.shape[-1])
    raw = (probabilities(alpha) - target).square().sum(-1)
    adjusted = (probabilities(candidates) - target[:, None, None]).square().sum(-1)
    return ((raw[:, None, None] - adjusted) / 2).clamp(-1, 1)


def action(delta, mask, threshold):
    """Return the chosen modality, action index, and whether it is executed."""
    eligible = mask & (mask.sum(1, keepdim=True) > 1)
    # Action-major order gives half-retention priority on exact ties.
    best, index = delta.masked_fill(~eligible[..., None], -torch.inf).transpose(1, 2).flatten(1).max(1)
    return index % mask.shape[1], index // mask.shape[1], best > threshold


def select_alpha(alpha, candidates, delta, mask, threshold):
    modality, choice, selected = action(delta, mask, threshold)
    rows = torch.arange(len(mask), device=mask.device)
    weight = torch.ones_like(mask, dtype=alpha.dtype)
    weight[rows, modality] -= selected * (choice + 1) / 2
    return torch.where(selected[:, None], candidates[rows, modality, choice], alpha), weight


def reliability(evidence, weight, mask):
    total = evidence.sum(-1)
    k = evidence.shape[-1]
    return mask * weight * (k + total) / (k + weight * total)


class ConditionalChannel(nn.Module):
    def __init__(self, features, context, width, latent, variant):
        super().__init__()
        self.features, self.context, self.variant = features, context, variant
        self.task = nn.Embedding(3, 4)
        self.encoder = nn.Sequential(nn.Linear(features + context + 5, width), nn.ReLU(),
                                     nn.Linear(width, width), nn.ReLU())
        self.gaussian = nn.Linear(width, 2 * latent)
        self.capacity = nn.Linear(width, 1)
        self.predict = nn.Linear(latent + context + 4 + 2, 1)
        self.register_buffer('index', torch.arange(latent) + .5)
        self.register_buffer('matched_center', torch.full((3,), latent / 2))
        self.register_buffer('random_offsets', torch.linspace(-1, 1, 128))

    def distribution(self, features, context, task, generator):
        features = F.pad(features, (0, self.features - features.shape[-1]))
        context = F.pad(context.detach(), (0, self.context - context.shape[-1]))
        condition = torch.cat((context, self.task.weight[task].expand(*context.shape[:-1], 4)), -1)
        flag = torch.ones_like(features[..., :1])
        prior_input = torch.cat((torch.zeros_like(features), condition, torch.zeros_like(flag)), -1)
        posterior_input = torch.cat((features, condition, flag), -1)
        if self.variant == 'context_only':
            posterior_input = prior_input
        mu, logvar = self.gaussian(self.encoder(posterior_input)).chunk(2, -1)
        if self.variant in {'context_only', 'no_ib'}:
            return mu, None, condition, torch.zeros_like(mu[..., 0]), torch.zeros_like(mu[..., 0])

        hidden = self.encoder(prior_input)
        prior, prior_var = self.gaussian(hidden).chunk(2, -1)
        prior_var = prior_var.clamp(-8, 8)
        if self.variant in {'fixed_vib', 'rib'}:
            prior, prior_var = torch.zeros_like(mu), torch.zeros_like(logvar)
        prior_sd = (prior_var / 2).exp()
        delta = (mu - prior) / prior_sd
        ratio = logvar.clamp(-8, 8) - prior_var

        gate = torch.ones_like(mu)
        if self.variant.endswith('acrib') or self.variant == 'idab_crib':
            center = len(self.index) * self.capacity(hidden).sigmoid()
            if self.variant == 'fixed_acrib':
                center = self.matched_center[task]
            elif self.variant == 'random_acrib':
                draw = torch.randint(len(self.random_offsets), center.shape, device=mu.device, generator=generator)
                center = self.matched_center[task] + self.random_offsets[draw]
            gate = (center - self.index).sigmoid().expand_as(mu)
        if self.variant.endswith('acrib') and self.variant != 'unbounded_acrib':
            delta = 2 * delta.tanh()
            ratio = (logvar - prior_var).tanh()

        variance_delta = gate.square() * ratio.expm1()
        gated_ratio = variance_delta.clamp_min(-1 + 1e-7).log1p()
        gated_delta = gate * delta
        kl = (.5 * (gated_delta.square() + variance_delta - gated_ratio)).sum(-1)
        if self.variant == 'idab_crib':
            # Straight-through ordered hard truncation of the conditional residual.
            hard = (gate >= .5).to(gate)
            gate = gate + (hard - gate).detach()
            kl = (.5 * gate * (delta.square() + ratio.expm1() - ratio)).sum(-1)
            std = prior_sd * (ratio / 2).exp() * gate
            mean = (prior + prior_sd * delta) * gate
        else:
            std = prior_sd * (gated_ratio / 2).exp()
            mean = prior + prior_sd * gated_delta
        return mean, std, condition, kl, gate.square().sum(-1)

    def forward(self, features, context, task, valid, generator, probe):
        mean, std, condition, kl, capacity = self.distribution(features, context, task, generator)
        z = mean
        if std is not None:
            z = mean + std * torch.randn(mean.shape, device=mean.device, generator=generator)
        if probe == 'zero':
            z = torch.zeros_like(z)
        elif probe == 'shuffle':
            # Permute across valid rows of each modality or directed pair separately.
            z = z.clone()
            for column in range(z.shape[1]):
                rows = valid[:, column].nonzero().flatten()
                z[rows, column] = z[rows.roll(1), column]
        design = torch.cat((z, condition), -1)
        if task == 2:
            design = design.unsqueeze(-2).expand(*design.shape[:-1], 2, design.shape[-1])
            code = torch.eye(2, device=z.device).expand(*design.shape[:-2], 2, 2)
            design = torch.cat((design, code), -1)
        else:
            design = F.pad(design, (0, 2))
        return self.predict(design).squeeze(-1), kl, capacity


class TMCACRIB(TMCHCRD):
    def __init__(self, dims, classes, variant, width=128, latent=16):
        super().__init__(dims, classes, 'tmc', width, latent)
        m = len(dims)
        self.register_buffer('identity', torch.eye(m))
        self.observation = nn.Linear(width, 1)
        context = max(12 + 2*m, 6 + 4*classes + m*(classes + 5) + 2*m)
        self.channel = ConditionalChannel(3*width, context, 64, latent, variant)

    def risk(self, features, evidence, mask, alpha, candidates, generator=None, probe=None):
        p, entropy, uncertainty = self.distribution(evidence)
        log_strength = (1 + evidence).sum(-1).log()
        identity = self.identity.expand(len(mask), -1, -1)
        o_logit = self.observation(features).squeeze(-1)
        o = o_logit.sigmoid()
        a_context = torch.cat((torch.stack((o, entropy, uncertainty, log_strength), -1), identity), -1)
        a_logit, kl_a, cap_a = self.channel(features, a_context, 0, mask, generator, probe)
        a = a_logit.sigmoid()

        i, j = self.i, self.j
        pair = mask[:, i] & mask[:, j]
        incidence = self.pair_i + self.pair_j
        weights = (1 - incidence) * (mask * (1 - o))[:, None]
        support = weights.sum(-1, keepdim=True)
        ref = torch.where(support > EPS, (weights @ p) / support.clamp_min(EPS), 1 / self.classes)
        raw = js(p[:, i], p[:, j])
        ref_i, ref_j = js(p[:, i], ref), js(p[:, j], ref)
        source, target = torch.cat((i, j)), torch.cat((j, i))
        hs, ht = features[:, source], features[:, target]
        pair_features = torch.cat((hs, ht, (hs - ht).abs()), -1)
        scalars = [value[:, endpoint] for value in (o, a, entropy, uncertainty) for endpoint in (source, target)]
        scalars.extend((raw.repeat(1, 2), torch.cat((ref_i, ref_j), 1), torch.cat((ref_j, ref_i), 1),
                        (support.squeeze(-1) > EPS).repeat(1, 2)))
        pair_context = torch.cat((torch.stack(scalars, -1), identity[:, source], identity[:, target]), -1)
        directed, kl_c, cap_c = self.channel(pair_features, pair_context, 1, pair.repeat(1, 2), generator, probe)
        left, right = directed.chunk(2, 1)
        c_logits = torch.stack((torch.zeros_like(left), left, right), -1)
        pi = c_logits.softmax(-1)
        c = ((pi[..., 1] * pair) @ self.pair_i + (pi[..., 2] * pair) @ self.pair_j)
        c = c / (pair.float() @ incidence).clamp_min(1)

        local = torch.stack((o, a, c, entropy, uncertainty, log_strength), -1)
        global_views = torch.cat((p, torch.stack((uncertainty, o, a, c), -1)), -1) * mask[..., None]
        global_context = torch.cat((mask[..., None], global_views), -1).flatten(1)
        d_context = torch.cat((local, p, probabilities(alpha)[:, None].expand_as(p),
                               probabilities(candidates).flatten(2), global_context[:, None].expand(-1, mask.shape[1], -1),
                               identity, mask[:, None].expand_as(identity)), -1)
        multiple = mask & (mask.sum(1, keepdim=True) > 1)
        d_logits, kl_d, cap_d = self.channel(features, d_context, 2, multiple, generator, probe)
        return dict(o=o, a=a, c=c, pi=pi, delta=2*d_logits.sigmoid()-1, o_logit=o_logit, a_logit=a_logit,
                    c_logits=c_logits, d_logits=d_logits, kl=(kl_a, kl_c, kl_d),
                    cap=(cap_a, cap_c, cap_d), valid=(mask, pair.repeat(1, 2), multiple))

    def infer(self, inputs, mask, generator, samples=8, probe=None):
        h, e = self.evidence(inputs)
        alpha, candidates = counterfactual(e, mask)
        outputs = [self.risk(h, e, mask, alpha, candidates, generator, probe) for _ in range(samples)]
        result = {k: torch.stack([out[k] for out in outputs]).mean(0) for k in ('o', 'a', 'c', 'pi', 'delta')}
        for key in ('kl', 'cap'):
            result[key] = tuple(torch.stack([out[key][t] for out in outputs]).mean(0) for t in range(3))
        result.update(valid=outputs[0]['valid'], alpha=alpha, candidates=candidates, evidence=e)
        return result


def classification_loss(alpha, evidence, mask, y, view_labels, epoch):
    anneal = min(1., (epoch + 1) / 10)
    single = evidential_loss(1 + evidence.flatten(0, 1), view_labels.flatten(), anneal).view_as(mask)
    return evidential_loss(alpha, y, anneal).mean() + mean_valid(single, mask)


def risk_loss(model, output, strength, cross, ambiguity, delta):
    mask, directed_mask, multiple = output['valid']
    pair = directed_mask.chunk(2, 1)[0]
    bce = F.binary_cross_entropy_with_logits
    labels = cross[:, model.i].long() + 2 * cross[:, model.j].long()
    pair_ce = F.cross_entropy(output['c_logits'].flatten(0, 1), labels.flatten(), reduction='none').view_as(pair)
    losses = (mean_valid(bce(output['o_logit'], (strength > 0).float(), reduction='none'), mask)
              + mean_valid(bce(output['a_logit'], ambiguity, reduction='none'), mask)
              + mean_valid(pair_ce, pair)
              + mean_valid(bce(output['d_logits'], (1 + delta) / 2, reduction='none').mean(-1), multiple))
    values = output['kl']
    if model.channel.variant == 'rib':
        weights = (output['o'].detach(), (output['pi'][..., 1:].sum(-1).detach()).repeat(1, 2),
                   output['delta'].detach().abs().amax(-1))
        values = tuple(value / (1 + weight) for value, weight in zip(values, weights))
    a, c, d = values
    kl = mean_valid(a, mask) + mean_valid(sum(c.chunk(2, 1)), pair) + mean_valid(d, multiple)
    count = 1 + pair.any().int() + multiple.any().int()
    return losses / (1 + count), kl / count
