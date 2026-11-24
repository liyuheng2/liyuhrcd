"""Coalition-aware information bottlenecks and conflict-conservative fusion."""
from math import comb

import torch
from torch import nn
from torch.nn import functional as F

from baselines import evidential_loss
from tmc_hcrd import EPS, js


VARIANTS = ('cibf', 'fixed_ib', 'no_ib', 'full_context_ib', 'no_discount', 'ds_fusion', 'vib_ds')


def opinion(evidence):
    classes = evidence.shape[-1]
    strength = classes + evidence.sum(-1, keepdim=True)
    return evidence / strength, classes / strength


def conjunctive(belief, uncertainty, ds=False):
    """Combine on the penultimate axis. Vacuous opinions act as the identity."""
    ignorance = uncertainty.prod(-2)
    mass = ((belief + uncertainty).prod(-2) - ignorance).clamp_min(0)
    conflict = (1 - ignorance - mass.sum(-1, keepdim=True)).clamp(0, 1)
    if ds:
        normalizer = (ignorance + mass.sum(-1, keepdim=True)).clamp_min(EPS)
        mass, uncertainty = mass / normalizer, ignorance / normalizer
    else:
        uncertainty = (1 - mass.sum(-1, keepdim=True)).clamp(0, 1)
    classes = mass.shape[-1]
    return dict(p=mass + uncertainty / classes, u=uncertainty.squeeze(-1),
                conflict=conflict.squeeze(-1), alpha=1 + classes * mass / uncertainty.clamp_min(EPS))


def fuse_evidence(evidence, discount, ds=False):
    belief, uncertainty = opinion(evidence)
    discount = discount[..., None].to(evidence.dtype)
    return conjunctive(discount * belief, uncertainty + (1 - discount) * (1 - uncertainty), ds)


def evidential_objective(fused, single, labels, view_labels, mask, anneal):
    batch, views, classes = single.shape
    losses = evidential_loss(single.reshape(-1, classes), view_labels.flatten(), anneal).view(batch, views)
    return (evidential_loss(fused, labels, anneal) + _mean_views(losses, mask)).mean()


def _mean_views(values, mask):
    return (values * mask).sum(1) / mask.sum(1).clamp_min(1)


class TMC_CIBF(nn.Module):
    def __init__(self, dims, classes, width=128, latent=16, variant='cibf'):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'Unknown CIBF variant: {variant}')
        self.variant = variant
        self.project = nn.ModuleList(nn.Sequential(nn.Linear(dim, width), nn.LayerNorm(width)) for dim in dims)
        self.posterior = nn.ModuleList(nn.Linear(width, 2 * latent) for _ in dims)
        self.evidence = nn.ModuleList(nn.Linear(latent, classes) for _ in dims)
        views = len(dims)
        self.smce = nn.Sequential(nn.Linear(3 * classes + 4 + 2 * views, 64), nn.GELU(),
                                  nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1), nn.Tanh())
        bits = 1 << torch.arange(views)
        subsets = (torch.arange(1 << views)[:, None] & bits).bool()
        coalition, view = (~subsets).nonzero(as_tuple=True)
        size = subsets[coalition].sum(1)
        weights = torch.zeros(views + 1, views)
        for n in range(1, views + 1):
            weights[n, :n] = torch.tensor([1 / (n * comb(n - 1, s)) for s in range(n)])
        for key, value in dict(bits=bits, subsets=subsets, before=coalition, view=view,
                               after=coalition + bits[view], subset_size=size, shapley=weights,
                               identity=F.one_hot(view, views).float()).items():
            self.register_buffer(key, value, persistent=False)

    def encode_evidence(self, latent):
        return torch.stack([F.softplus(head(z)) for head, z in zip(self.evidence, latent.unbind(1))], 1)

    def contributions(self, deterministic, mask):
        with torch.no_grad():
            view_mask = self.subsets[None] & mask[:, None]
            opinions = fuse_evidence(deterministic[:, None], view_mask)
            p, u, conflict = opinions['p'], opinions['u'], opinions['conflict']
            solo, before, after = p[:, self.bits[self.view]], p[:, self.before], p[:, self.after]
            facts = torch.stack((u[:, self.bits[self.view]], u[:, self.before], js(solo, before),
                                 conflict[:, self.before]), -1)
            identities = self.identity.expand(len(p), -1, -1)
            members = self.subsets[self.before].expand(len(p), -1, -1)
            features = torch.cat((solo, before, after, facts, identities, members), -1)
            available = (mask * self.bits).sum(1)
            valid = (self.after[None] & available[:, None]) == self.after
            current = self.after[None] == available[:, None]
            weight = self.shapley[mask.sum(1, keepdim=True), self.subset_size] * valid
        edge = self.smce(features).squeeze(-1)
        return dict(edge=edge, context=(edge * current) @ self.identity,
                    phi=(edge * weight) @ self.identity, weight=weight, valid=valid, current=current,
                    before=before, after=after)

    def forward(self, inputs, mask, warmup=False):
        posterior = torch.stack([head(project(x)) for project, head, x in
                                 zip(self.project, self.posterior, inputs)], 1)
        mean, logvar = posterior.chunk(2, -1)
        logvar = logvar.clamp(-10, 8)
        deterministic = self.encode_evidence(mean)
        contribution = self.contributions(deterministic, mask)
        context = contribution['context']
        phi = context if self.variant == 'full_context_ib' else contribution['phi']
        discount = mask.to(mean.dtype)
        if not warmup and self.variant not in {'no_discount', 'vib_ds'}:
            discount = discount * (1 - (-context).relu()).detach()
        ds = self.variant in {'ds_fusion', 'vib_ds'}
        result = fuse_evidence(deterministic, discount, ds)
        result.update(single_alpha=1 + deterministic, contribution=contribution, discount=discount,
                      rate=.5 * (mean.square() + logvar.exp() - logvar - 1).mean(-1),
                      compression=(1 - .5 * phi.relu() + (-phi).relu()).detach())
        if self.training:
            sampled = self.encode_evidence(mean + (.5 * logvar).exp() * torch.randn_like(mean))
            result['sample_alpha'] = fuse_evidence(sampled, discount, ds)['alpha']
            result['sample_single_alpha'] = 1 + sampled
        return result

    def loss(self, out, labels, view_labels, mask, anneal, beta, balance, warmup):
        task = .5 * (evidential_objective(out['alpha'], out['single_alpha'], labels, view_labels, mask, anneal)
                     + evidential_objective(out['sample_alpha'], out['sample_single_alpha'], labels,
                                            view_labels, mask, anneal))
        rate = out['rate']
        if not warmup and self.variant not in {'fixed_ib', 'vib_ds'}:
            rate = rate * out['compression']
        ib = _mean_views(rate, mask).mean()
        c = out['contribution']
        contribution = (((c['edge'] - self.targets(c, labels)).square() * c['weight']).sum(1)
                        / mask.sum(1).clamp_min(1)).mean()
        weighted_ib = 0 if self.variant == 'no_ib' else beta * ib
        weighted_contribution = 0 if warmup or self.variant == 'vib_ds' else balance * contribution
        return task + weighted_ib + weighted_contribution, (task, ib, contribution)

    @staticmethod
    def targets(contribution, labels):
        truth = F.one_hot(labels, contribution['before'].shape[-1])[:, None]
        before = (contribution['before'] - truth).square().sum(-1)
        after = (contribution['after'] - truth).square().sum(-1)
        return .5 * (before - after)


@torch.inference_mode()
def predict(model, inputs, mask):
    """Predict from standardized views using a model in evaluation mode."""
    if model.training:
        raise ValueError('Call model.eval() before predict()')
    out = model(inputs, mask)
    predictable = mask.any(1)
    return dict(prediction=out['p'].argmax(1).masked_fill(~predictable, -1), predictable=predictable,
                probability=out['p'], uncertainty=out['u'], conflict=out['conflict'],
                retention=out['discount'], contribution=out['contribution']['context'])
