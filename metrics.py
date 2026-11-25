"""Classification, calibration, error detection and tie-aware selective risk."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def detection(label, score):
    label, score = np.asarray(label), np.asarray(score)
    if len(np.unique(label)) < 2:
        return {'auroc': None, 'auprc': None}
    return {'auroc': float(roc_auc_score(label, score)),
            'auprc': float(average_precision_score(label, score))}


def calibration(confidence, target, bins=15):
    bucket = np.minimum((confidence * bins).astype(int), bins - 1)
    counts = np.bincount(bucket, minlength=bins)
    difference = np.bincount(bucket, weights=confidence - target, minlength=bins)
    return float(np.abs(difference).sum() / counts.sum())


def risk_coverage(correct, score):
    sort = np.argsort(-score, kind='stable')
    ordered, errors = score[sort], (~correct[sort]).astype(float)
    bounds = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1, len(sort)]
    lengths = np.diff(bounds)
    if (lengths > 1).any():
        errors = np.repeat(np.add.reduceat(errors, bounds[:-1]) / lengths, lengths)
    n = np.arange(1, len(sort) + 1)
    return n / len(sort), np.cumsum(errors) / n


def summarize(y, p, reliability, threshold=None):
    prediction, confidence = p.argmax(1), p.max(1)
    correct = prediction == y
    coverage_grid, risk = risk_coverage(correct, reliability)
    err = detection(~correct, -reliability)
    if np.array_equal(reliability, confidence):
        maxprob_coverage, maxprob_risk, maxprob_err = coverage_grid, risk, err
    else:
        maxprob_coverage, maxprob_risk = risk_coverage(correct, confidence)
        maxprob_err = detection(~correct, -confidence)
    onehot = np.eye(p.shape[1], dtype=np.float32)[y]
    stats = {
        'n': int(len(y)),
        'accuracy': float(correct.mean()),
        'macro_f1': float(f1_score(y, prediction, labels=np.arange(p.shape[1]), average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y, prediction, average='weighted', zero_division=0)),
        'nll': float(-np.log(p[np.arange(len(y)), y].clip(1e-7, 1)).mean()),
        'brier': float(np.square(p - onehot).sum(1).mean()),
        'ece15': calibration(confidence, correct.astype(float)),
        'classwise_ece15': float(np.mean([calibration(p[:, k], onehot[:, k]) for k in range(p.shape[1])])),
        'aurc': float(risk.mean()),
        'error_auroc': err['auroc'],
        'error_auprc': err['auprc'],
        'maxprob_aurc': float(maxprob_risk.mean()),
        'maxprob_error_auroc': maxprob_err['auroc'],
        'maxprob_error_auprc': maxprob_err['auprc'],
    }
    if threshold is not None:
        keep = reliability >= threshold
        stats.update(coverage=float(keep.mean()),
                     selective_accuracy=float(correct[keep].mean()) if keep.any() else None)
    for coverage in (0.5, 0.8, 0.9, 1.0):
        index = min(len(risk) - 1, int(np.ceil(coverage * len(risk))) - 1)
        stats[f'acc_at_{int(coverage * 100)}pct'] = float(1 - risk[index])
    return stats, np.column_stack((coverage_grid, risk, 1 - risk, maxprob_coverage, maxprob_risk))


def regression(target, prediction):
    if not len(target):
        return {'mae': None, 'spearman': None}
    correlation = None
    if len(target) > 1 and np.ptp(target) > 0 and np.ptp(prediction) > 0:
        correlation = float(spearmanr(target, prediction).statistic)
    return {'mae': float(np.abs(target - prediction).mean()), 'spearman': correlation}

