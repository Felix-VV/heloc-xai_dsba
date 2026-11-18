"""Predictive metrics, calibration summaries and paired bootstrap."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPS = 1e-12


def score_all(y, p) -> dict:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return {"auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p), "logloss": log_loss(y, p)}


def calibration_table(y, p, n_bins=10) -> dict:
    """Quantile-binned reliability table plus logistic recalibration (intercept, slope).

    Slope < 1 indicates over-dispersed (too extreme) probabilities; the intercept
    is estimated jointly with the slope. `citl` is mean(y) - mean(p).
    """
    y = np.asarray(y); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    bins = [{"bin": int(b), "n": int((idx == b).sum()), "mean_pred": float(p[idx == b].mean()),
             "obs_rate": float(y[idx == b].mean())} for b in np.unique(idx)]
    z = np.log(p / (1 - p))

    def nll(theta):
        a, b = theta
        eta = a + b * z
        return np.sum(np.logaddexp(0, eta) - y * eta)

    a, b = minimize(nll, x0=[0.0, 1.0], method="BFGS").x
    return {"bins": bins, "recal_intercept": float(a), "recal_slope": float(b),
            "citl": float(y.mean() - p.mean())}


def paired_bootstrap(y, preds: dict, n_boot: int, seed: int) -> dict:
    """Percentile CIs for each model's AUC/Brier and for pairwise differences.

    Resamples test cases only: the fitted models are held fixed, so these intervals
    ignore training-set variability (see the repeated-split study for that).
    """
    rng = np.random.default_rng(seed)
    names = list(preds)
    y = np.asarray(y)
    n = len(y)
    stats = {f"{m}_{s}": [] for m in names for s in ("auc", "brier")}
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    for a, b in pairs:
        for s in ("auc", "brier"):
            stats[f"{a}-{b}_{s}"] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        vals = {}
        for m in names:
            pb = preds[m][idx]
            vals[(m, "auc")] = roc_auc_score(yb, pb)
            vals[(m, "brier")] = np.mean((pb - yb) ** 2)
            stats[f"{m}_auc"].append(vals[(m, "auc")])
            stats[f"{m}_brier"].append(vals[(m, "brier")])
        for a, b in pairs:
            for s in ("auc", "brier"):
                stats[f"{a}-{b}_{s}"].append(vals[(a, s)] - vals[(b, s)])
    out = {}
    for k, v in stats.items():
        v = np.asarray(v)
        out[k] = {"lo": float(np.quantile(v, 0.025)), "hi": float(np.quantile(v, 0.975)),
                  "sd": float(v.std(ddof=1)), "n": int(len(v))}
    return out
