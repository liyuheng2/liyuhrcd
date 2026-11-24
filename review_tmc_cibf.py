"""Post-hoc diagnostics using the original checkpoints, feature splits and perturbations."""
from pathlib import Path

from scipy.optimize import minimize_scalar
import torch
from torch.nn import functional as F

from baselines import Baseline
from data import condition, load_data
from main import atomic_json, digest_cache
from metrics import calibration
from summarize_tmc_cibf import read_cells
from tmc_cibf import TMC_CIBF, fuse_evidence


def brier(p, y):
    return (p - F.one_hot(y, p.shape[-1])).square().sum(-1)


@torch.inference_mode()
def fit_temperature(model, validation):
    views, mask, _, _, _ = condition(validation, 'clean')
    labels = validation[2]
    present = mask.any(1).nonzero().flatten()
    scores = torch.cat([model([v[idx] for v in views], mask[idx])['p']
                        for idx in present.split(128)]).double().log()

    def objective(temperature):
        return F.cross_entropy(scores / temperature, labels[present]).item()

    fit = minimize_scalar(objective, method='bounded', bounds=(0.1, 10.0))
    if not fit.success:
        raise ValueError('Temperature fitting failed')
    return float(fit.x)


def calibrated_metrics(p, labels, temperature):
    q = (p.double().log() / temperature).softmax(-1)
    confidence, prediction = q.max(-1)
    if not torch.equal(prediction, p.argmax(-1)):
        raise ValueError('Positive temperature changed a prediction')
    return dict(temperature=temperature, brier=float(brier(q, labels).mean()),
                nll=float(F.nll_loss(q.log(), labels)),
                ece15=calibration(confidence.numpy(), prediction.eq(labels).numpy()))


@torch.inference_mode()
def measure(model, baseline, vib_ds, test, classes, condition_name, temperatures):
    views, mask, _, _, _ = condition(test, condition_name)
    y = test[2]
    chunks = []
    for idx in torch.arange(len(y)).split(128):
        x, m, labels = [v[idx] for v in views], mask[idx], y[idx]
        out = model(x, m)
        ref = baseline(x, m)['p']
        vib_p = vib_ds(x, m)['p']
        evidence = out['single_alpha'] - 1
        vanilla = fuse_evidence(evidence, m)['p']
        ds = fuse_evidence(evidence, out['discount'], ds=True)['p']
        p = out['p']
        conflict = out['conflict'][:, None]
        torch.testing.assert_close(p, (1 - conflict) * ds + conflict / classes, atol=2e-6, rtol=2e-6)

        # Best of keeping all or fully dropping one view, using the test label only for this oracle.
        unchanged = brier(vanilla, labels)
        candidates = [unchanged]
        for i in range(m.shape[1]):
            keep = m.clone()
            keep[:, i] = False
            dropped = brier(fuse_evidence(evidence, keep)['p'], labels)
            candidates.append(torch.where(keep.any(1), dropped, unchanged))
        risks = torch.stack(candidates, 1)
        oracle = risks.min(1).values

        delta = out['contribution']
        target = (model.targets(delta, labels) * delta['current']) @ model.identity
        minimum, worst = delta['context'].masked_fill(~m, float('inf')).min(1)
        action = torch.where(minimum < 0, 1 + worst, 0)
        hard_prediction = risks.gather(1, action[:, None]).squeeze(1)
        label_discount = m * (1 - (-target).relu())
        soft_oracle = brier(fuse_evidence(evidence, label_discount)['p'], labels)
        chunks.append(dict(p=p, base=ref, vib=vib_p, vanilla=vanilla, ds=ds, y=labels, oracle=oracle,
                           hard_prediction=hard_prediction, soft_oracle=soft_oracle,
                           target=target[m], value=delta['context'][m], retain=out['discount'][m],
                           omega=out['compression'][m]))
    v = {key: torch.cat([part[key] for part in chunks]) for key in chunks[0]}
    p, ref, y = v['p'], v['base'], v['y']
    harmful, flagged = v['target'] < 0, v['value'] < 0
    recall = (flagged & harmful).sum() / harmful.sum().clamp_min(1)
    predictions, baseline_predictions = p.argmax(-1), ref.argmax(-1)
    def confusion(prediction):
        return torch.bincount(y * classes + prediction, minlength=classes ** 2).reshape(classes, classes).tolist()
    return dict(samples=len(y), brier=float(brier(p, y).mean()), accuracy=float((predictions == y).float().mean()),
                base_brier=float(brier(ref, y).mean()), confidence=float(p.max(-1).values.mean()),
                vib_brier=float(brier(v['vib'], y).mean()),
                base_confidence=float(ref.max(-1).values.mean()), harmful_fraction=float(harmful.float().mean()),
                flagged_fraction=float(flagged.float().mean()), harmful_recall=float(recall),
                discount_over_10pct=float((v['retain'] < .9).float().mean()),
                omega_abs_deviation=float((v['omega'] - 1).abs().mean()),
                brier_delta_discount=float((brier(p, y) - brier(v['vanilla'], y)).mean()),
                classification_flip_discount=float((predictions != v['vanilla'].argmax(-1)).float().mean()),
                brier_delta_fusion=float((brier(p, y) - brier(v['ds'], y)).mean()),
                classification_flip_fusion=float((predictions != v['ds'].argmax(-1)).float().mean()),
                hard_prediction_gain=float((brier(v['vanilla'], y) - v['hard_prediction']).mean()),
                label_soft_gain=float((brier(v['vanilla'], y) - v['soft_oracle']).mean()),
                oracle_gain_over_no_discount=float((brier(v['vanilla'], y) - v['oracle']).mean()),
                calibrated_cibf=calibrated_metrics(p, y, temperatures[0]),
                calibrated_tmc_aug=calibrated_metrics(ref, y, temperatures[1]),
                calibrated_vib_ds=calibrated_metrics(v['vib'], y, temperatures[2]),
                cibf_confusion=confusion(predictions), tmc_aug_confusion=confusion(baseline_predictions))


def main():
    torch.set_num_threads(1)
    root = Path('runs/tmc_cibf')
    rows = []
    previous = None
    for cell in read_cells(root):
        run = cell['run']
        dataset, fold, seed = run['dataset'], run['fold'], run['seed']
        group = f'iemocap_loso{fold}' if fold else dataset
        if group != previous:
            cache = Path('data/crib_iemocap') / f'loso{fold}' if fold else Path('data/extended')
            fingerprint = digest_cache(cache, dataset)
            splits, classes = load_data(cache, dataset)
            previous = group
        if fingerprint != run['sha256']:
            raise ValueError(f'Data fingerprints changed for {group}')
        weights = torch.load(root / group / str(seed) / 'models.pt', weights_only=True, map_location='cpu')
        dims = [x.shape[1] for x in splits['train'][0]]
        model = TMC_CIBF(dims, classes, run['settings']['width'], run['settings']['latent']).eval()
        baseline = Baseline(dims, classes, 'tmc', len(splits['train'][2]), run['settings']['width']).eval()
        vib_ds = TMC_CIBF(dims, classes, run['settings']['width'], run['settings']['latent'], 'vib_ds').eval()
        model.load_state_dict(weights['cibf'])
        baseline.load_state_dict(weights['tmc_aug'])
        vib_ds.load_state_dict(weights['vib_ds'])
        temperatures = [fit_temperature(net, splits['val']) for net in (model, baseline, vib_ds)]
        for name in ('clean', 'conflict'):
            diagnostic = measure(model, baseline, vib_ds, splits['test'], classes, name, temperatures)
            original = cell['metrics'][name]['methods']
            for key, method in [('brier', 'cibf'), ('base_brier', 'tmc_aug'), ('vib_brier', 'vib_ds')]:
                if abs(diagnostic[key] - original[method]['brier']) > 2e-6:
                    raise ValueError(f'Cannot reproduce Brier for {group}/{seed}/{name}/{method}')
            if abs(diagnostic['accuracy'] - original['cibf']['accuracy']) > 2e-6:
                raise ValueError(f'Cannot reproduce CIBF accuracy for {group}/{seed}/{name}')
            rows.append(dict(dataset=dataset, fold=fold, seed=seed, condition=name, **diagnostic))
    atomic_json(Path('results/tmc_cibf/review_diagnostics.json'), rows)
    print(f'Verified {len(rows)} checkpoint-condition evaluations.')


if __name__ == '__main__':
    main()
