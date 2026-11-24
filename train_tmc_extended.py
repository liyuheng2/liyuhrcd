"""Train the fixed-setting TMC comparisons, retaining teachers and best checkpoints."""
from argparse import ArgumentParser
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
from time import perf_counter

from filelock import FileLock
import torch
from torch import nn
from torch.nn import functional as F

from baselines import Baseline
from data import load_data, training_batch
from main import Settings, atomic_json, batches, digest_cache, linear_macs
from tmc_hcrd import TMCHCRD, mean_valid
from validate_tmc import DATASETS, SEEDS, fit


PRIMARY = ('tmc', 'capacity', 'simple', 'only_o', 'hcrd_equal', 'fixed_equal', 'dynamic_equal', 'full')
EXTERNAL = ('concat', 'late', 'gate', 'etmc')
IB = ('zero_equal', 'mean_equal', 'shuffle_equal', 'reverse_equal')
PROTOCOLS = ('tmc_clean', 'simple_clean', 'tmc_no_dropout', 'full_no_dropout')
METHODS = (*PRIMARY, *EXTERNAL, *IB, *PROTOCOLS)
ROOT = Path('runs/tmc_extended')


class ExtendedTMC(TMCHCRD):
    def __init__(self, dims, classes, method, settings):
        self.variant = method
        base = method.removesuffix('_clean').removesuffix('_no_dropout')
        super().__init__(dims, classes, 'tmc' if base == 'capacity' else base, settings.width, settings.latent)
        self.reference_weights = None
        if method == 'capacity':
            risk_parameters = 27 + 17 * settings.latent + 13 + 64 * (12 + 2 * settings.latent) + settings.width
            mid = max(1, round((risk_parameters / len(dims) - settings.width) / (2 * settings.width + 1)))
            self.extra = nn.ModuleList(nn.Sequential(nn.Linear(settings.width, mid), nn.GELU(),
                                       nn.Linear(mid, settings.width)) for _ in dims)
        if method == 'only_o':
            del self.rib

    def evidence(self, inputs):
        if self.variant != 'capacity':
            return super().evidence(inputs)
        features = torch.stack([enc(x) for enc, x in zip(self.encoders, inputs)], dim=1)
        evidence = [head(z + layer(z)) for head, layer, z in zip(self.heads, self.extra, features.unbind(1))]
        return features, F.softplus(torch.stack(evidence, dim=1))

    def forward(self, inputs, mask, warmup=False, risk_only=False):
        if self.variant != 'only_o' or warmup:
            return super().forward(inputs, mask, warmup, risk_only)
        from tmc_hcrd import fuse
        features, evidence = self.evidence(inputs)
        _, _, u = self.distribution(evidence)
        logit = self.observation(features.detach() if risk_only else features).squeeze(-1)
        o = logit.sigmoid()
        discount = (1 - o) * mask
        alpha = fuse(discount[..., None] * evidence)
        total = alpha.sum(-1)
        return dict(p=alpha / total[:, None], alpha=alpha, score=1 - self.classes / total,
                    single_alpha=1 + evidence, u=u, discount=discount, o=o, o_logit=logit)

    def risk_loss(self, result, mask, strength, refs, beta):
        if self.variant == 'only_o':
            return mean_valid(F.binary_cross_entropy_with_logits(result['o_logit'], strength,
                                                                 reduction='none'), mask)
        return super().risk_loss(result, mask, strength, refs, 0 if self.variant == 'zero_equal' else beta)

    def kl_weights(self, result, strength, normal, entropy_clean, mask):
        if self.variant == 'mean_equal':
            return self.reference_weights
        wa, wc = super().kl_weights(result, strength, normal, entropy_clean, mask)
        if self.variant == 'reverse_equal':
            return wa.reciprocal(), wc.reciprocal()
        if self.variant == 'shuffle_equal':
            def permute(weight, valid):
                shuffled = weight.clone()
                pool = weight[valid]
                shuffled[valid] = pool[torch.randperm(len(pool), device=weight.device)]
                return shuffled
            return permute(wa, mask), permute(wc, result['pair_mask'])
        return wa, wc


class ExternalTMC(Baseline):
    uses_risk = False

    def forward(self, inputs, mask, **kwargs):
        return super().forward(inputs, mask)

    def training_loss(self, result, y, mask, epoch):
        return self.loss(result, y, mask, None, epoch)


def make_model(data, classes, method, cfg):
    dims = [x.shape[1] for x in data['train'][0]]
    if method in EXTERNAL:
        return ExternalTMC(dims, classes, method, 0, cfg.width)
    return ExtendedTMC(dims, classes, method, cfg)


@torch.inference_mode()
def mean_weights(model, teacher, data, seed, batch):
    rng = torch.Generator().manual_seed(seed + 992)
    totals, counts = torch.zeros(2), torch.zeros(2)
    for *clean, mask, _, _ in batches(data, batch):
        current, mask, strength = training_batch(clean, mask, rng)
        out = model(current, mask)
        _, normal, h = model.references(teacher, clean, current)
        weights = model.dynamic_weights(out, strength, normal, h)
        for i, (weight, valid) in enumerate(zip(weights, (mask, out['pair_mask']))):
            totals[i] += weight[valid].sum()
            counts[i] += valid.sum()
    return (totals / counts.clamp_min(1)).tolist()


def train_cell(data, classes, name, method, seed, cfg, root, fingerprint):
    directory = root / name / method / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / 'result.json'
    run = dict(protocol='extended-v1', dataset=name, method=method, seed=seed,
               settings=asdict(cfg), fingerprint=fingerprint, device='cpu', torch=str(torch.__version__))
    with FileLock(str(directory / '.lock')):
        if report.exists():
            if json.loads(report.read_text(encoding='utf-8'))['run'] != run:
                raise ValueError(f'Incompatible training settings in {directory}')
            return
        torch.manual_seed(seed)
        model = make_model(data, classes, method, cfg)
        if method == 'mean_equal':
            reference = make_model(data, classes, 'dynamic_equal', cfg).eval()
            folder = root / name / 'dynamic_equal' / str(seed)
            reference.load_state_dict(torch.load(folder / 'model.pt', weights_only=True))
            teacher = make_model(data, classes, 'dynamic_equal', cfg).eval()
            teacher.load_state_dict(torch.load(folder / 'teacher.pt', weights_only=True))
            model.reference_weights = mean_weights(reference, teacher, data['train'], seed, cfg.batch)
        best, selected, epoch = float('inf'), 0, 0
        state = None

        @torch.no_grad()
        def select(candidate):
            nonlocal best, selected, epoch, state
            epoch += 1
            candidate.eval()
            losses = []
            for *x, m, y, _ in batches(data['val'], cfg.batch):
                p = candidate(x, m)['p']
                losses.append(-p[torch.arange(len(y)), y].clamp_min(1e-8).log())
            score = torch.cat(losses).mean().item()
            if score < best:
                best, selected = score, epoch
                state = {k: v.clone() for k, v in candidate.state_dict().items()}
            return epoch - selected >= 10

        print(f'{root.name} {name} {method} seed={seed}', flush=True)
        start = perf_counter()
        teacher = fit(model, data['train'], seed, cfg, torch.device('cpu'), select,
                      clean_only=method.endswith('_clean'), modality_dropout=not method.endswith('_no_dropout'))
        seconds = perf_counter() - start
        model.load_state_dict(state)
        model.eval()
        torch.save(state, directory / 'model.pt')
        if teacher is not None:
            torch.save(teacher.state_dict(), directory / 'teacher.pt')
        atomic_json(report, dict(run=run, selected_epoch=selected, joint_epochs=epoch, val_nll=best,
                                 reference_weights=getattr(model, 'reference_weights', None),
                                 train_seconds=seconds, parameters=sum(p.numel() for p in model.parameters()),
                                 linear_macs=linear_macs(model, *data['val'][:2], torch.device('cpu'))))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument('--fold', type=int, choices=range(1, 6))
    parser.add_argument('--feature-dir', type=Path, default=Path('data/extended'))
    parser.add_argument('--run-dir', type=Path, default=ROOT)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = Settings()
    if args.smoke:
        cfg = replace(cfg, clean_epochs=1, risk_epochs=1, joint_epochs=1)
        args.run_dir /= 'smoke'
    torch.set_num_threads(1)
    if args.fold:
        args.feature_dir /= f'loso{args.fold}'
        args.run_dir /= f'loso{args.fold}'
        args.datasets = ['iemocap']
    for name in args.datasets:
        data, classes = load_data(args.feature_dir, name)
        fingerprint = digest_cache(args.feature_dir, name)
        for seed, method in product(args.seeds, args.methods):
            train_cell(data, classes, name, method, seed, cfg, args.run_dir, fingerprint)


if __name__ == '__main__':
    main()
