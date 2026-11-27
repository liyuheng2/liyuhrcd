"""Fixed-feature ports of CDPR and SeD-UD for the matched EC-RIB protocol."""
from itertools import combinations
from math import log

import torch
from torch import nn
from torch.nn import functional as F

from model import entropy, masked_mean


def projection(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.GELU())


def classification(result, labels, mask):
    logits = result['logits']
    target = labels[:, None, None].expand(-1, logits.shape[1], 1)
    per_view = -logits.log_softmax(-1).gather(-1, target).squeeze(-1)
    return F.cross_entropy(result['fused_logits'], labels) + masked_mean(per_view, mask)


class CDPRFeatures(nn.Module):
    """Consensus and discrepancy pathways with semantic and statistical conflict."""

    def __init__(self, dims, classes, width):
        super().__init__()
        count = len(dims)
        self.pairs = tuple(combinations(range(count), 2))
        self.adapters = nn.ModuleList(projection(dim, width) for dim in dims)
        self.private = nn.ModuleList(projection(width, width) for _ in dims)
        self.shared = projection(width, width)
        self.raw = nn.Linear(count * width, width)
        self.agreement = projection(len(self.pairs) * width, width)
        self.consensus_scale = nn.Parameter(torch.zeros(()))
        self.intuition_norm = nn.LayerNorm(width)
        self.discrepancy = projection(count * width, width)
        self.prototype = nn.Parameter(torch.randn(width))
        self.statistic = nn.Linear(count + 1, 1)
        self.uncertainty = projection(count, width)
        self.trust = nn.Linear(width, count)
        self.gate_bias = nn.Parameter(torch.tensor(-1.0))
        self.heads = nn.ModuleList(nn.Linear(width, classes) for _ in dims)
        self.reason_head = nn.Linear(width, classes)
        self.final_head = nn.Linear(width, classes)

    def forward(self, inputs, mask):
        present = mask[..., None]
        weights = mask / mask.sum(1, keepdim=True)
        features = torch.stack([adapter(x) for adapter, x in zip(self.adapters, inputs)], 1) * present
        private = torch.stack([adapter(z) for adapter, z in zip(self.private, features.unbind(1))], 1) * present
        shared = self.shared(features) * present
        agreement = torch.cat([shared[:, i] * shared[:, j] for i, j in self.pairs], -1)
        intuition = self.intuition_norm(self.raw(features.flatten(1)) + self.consensus_scale * self.agreement(agreement))
        center = (private * weights[..., None]).sum(1, keepdim=True)
        shape = self.discrepancy(((private - center).abs() * present).flatten(1))
        logits = torch.stack([head(z) for head, z in zip(self.heads, private.unbind(1))], 1)
        probs = logits.softmax(-1)
        average = (probs * weights[..., None]).sum(1, keepdim=True)
        js = ((probs * (probs.clamp_min(1e-7).log() - average.clamp_min(1e-7).log())).sum(-1)
              * weights).sum(1, keepdim=True).detach() / log(mask.shape[1])
        uncertainty = entropy(probs) * mask
        semantic_energy = (F.normalize(shape, dim=-1) * F.normalize(self.prototype, dim=0)).sum(-1, keepdim=True)
        level = (semantic_energy + self.statistic(torch.cat((js, uncertainty), 1))).sigmoid()
        reflection = level * shape
        alpha = self.trust(reflection + self.uncertainty(uncertainty)).masked_fill(~mask, -torch.inf).softmax(1)
        reasoning = (private * alpha[..., None]).sum(1)
        gate = (reflection.norm(dim=-1, keepdim=True).tanh() + js + self.gate_bias).sigmoid()
        fused_logits = self.final_head(torch.lerp(intuition, reasoning, gate))
        p = fused_logits.softmax(-1)
        return dict(p=p, score=p.max(-1).values, alpha=alpha, logits=logits, fused_logits=fused_logits,
                    reasoning=self.reason_head(reasoning), private=private, shared=shared)

    def loss(self, result, labels, mask, rows, epoch):
        private, shared = result['private'], result['shared']
        orthogonality = masked_mean(F.cosine_similarity(private, shared, dim=-1).square(), mask)
        consistency = shared.new_zeros(())
        for left, right in self.pairs:
            both = mask[:, left] & mask[:, right]
            consistency = consistency + masked_mean((shared[:, left] - shared[:, right]).square().mean(1), both)
        return (classification(result, labels, mask) + 0.3 * F.cross_entropy(result['reasoning'], labels)
                + 0.1 * (orthogonality + consistency / len(self.pairs)))


class AdaptiveCompression(nn.Module):
    """Rank latent units from clean training, then select a sample-specific prefix."""

    def __init__(self, width):
        super().__init__()
        self.compress = nn.Linear(width, width)
        self.reconstruct = nn.Linear(width, width)
        self.dimension = nn.Linear(1, 1)
        self.register_buffer('rank', torch.arange(width))
        self.register_buffer('importance', torch.zeros(width), persistent=False)
        self.selection = False

    def get_extra_state(self):
        return self.selection

    def set_extra_state(self, state):
        self.selection = bool(state)

    @torch.no_grad()
    def record_importance(self):
        weight = self.compress.weight
        if weight.grad is not None:
            self.importance.add_((weight * weight.grad).abs().sum(1))

    @torch.no_grad()
    def enable_selection(self):
        self.rank.copy_(self.importance.argsort(descending=True))
        self.selection = True

    def forward(self, values, influence):
        encoded = F.relu(self.compress(values))
        if not self.selection:
            return self.reconstruct(encoded)
        width = self.rank.numel()
        capacity = width / 4 + (3 * width / 4) * self.dimension(influence).sigmoid()
        positions = torch.arange(width, device=values.device)
        soft = ((capacity - positions) * 2).sigmoid()
        hard = (positions < capacity.round()).to(soft.dtype)
        ranked = hard.detach() + (soft - soft.detach())
        return self.reconstruct(encoded * ranked.scatter(-1, self.rank.expand_as(ranked), ranked))


class SeDUDFeatures(nn.Module):
    """Hierarchical redundancy compression, gated fusion and fused denoising."""

    def __init__(self, dims, classes, width, text_index):
        super().__init__()
        self.text_index = text_index
        self.adapters = nn.ModuleList(projection(dim, width) for dim in dims)
        self.redundancy = nn.ModuleList(nn.Linear(2 * width, 1) for _ in dims)
        self.per_view = nn.ModuleList(AdaptiveCompression(width) for _ in dims)
        self.gates = nn.ModuleList(nn.Linear(2 * width, 1) for _ in dims)
        self.norm = nn.LayerNorm(width)
        self.noise = nn.Linear(width, width)
        self.denoise = AdaptiveCompression(width)
        self.head = nn.Linear(width, classes)

    def record_importance(self):
        for block in (*self.per_view, self.denoise):
            block.record_importance()

    def enable_selection(self):
        for block in (*self.per_view, self.denoise):
            block.enable_selection()

    def forward(self, inputs, mask):
        present = mask[..., None]
        z = torch.stack([adapter(x) for adapter, x in zip(self.adapters, inputs)], 1) * present
        peers = mask.sum(1, keepdim=True) - mask.to(z.dtype)
        other = (z.sum(1, keepdim=True) - z) / peers.clamp_min(1)[..., None]
        influences = [layer(torch.cat((primary, aux), 1)).sigmoid() for layer, primary, aux in
                      zip(self.redundancy, z.unbind(1), other.unbind(1))]
        compressed = torch.stack([block(values, influence) for block, values, influence in
                                  zip(self.per_view, z.unbind(1), influences)], 1) * present
        overall = compressed.sum(1) / mask.sum(1, keepdim=True)
        text = torch.where(present[:, self.text_index], compressed[:, self.text_index], overall)
        scores = torch.cat([gate(torch.cat((text, values), 1)) for gate, values in
                            zip(self.gates, compressed.unbind(1))], 1)
        alpha = scores.masked_fill(~mask, -torch.inf).softmax(1)
        fused = self.norm(text + (compressed * alpha[..., None]).sum(1))
        noise = (self.noise(fused).sigmoid() * fused.abs()).mean(1, keepdim=True).sigmoid()
        denoised = self.denoise(fused, noise)
        logits = self.head(compressed)
        fused_logits = self.head(denoised)
        p = fused_logits.softmax(-1)
        return dict(p=p, score=p.max(-1).values, alpha=alpha, logits=logits, fused_logits=fused_logits,
                    original=z, compressed=compressed, fused=fused, denoised=denoised)

    def loss(self, result, labels, mask, rows, epoch):
        view_error = (result['compressed'] - result['original'].detach()).square().mean(-1)
        fused_error = F.mse_loss(result['denoised'], result['fused'].detach())
        return classification(result, labels, mask) + 0.1 * (masked_mean(view_error, mask) + fused_error)
