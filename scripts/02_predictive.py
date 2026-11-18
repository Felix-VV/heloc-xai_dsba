"""Model selection on shared CV folds, single final-test evaluation, split-variability study.

The final test set is used exactly once, after all three models are selected and refit.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd

from hxai.common import MODELS, RESULTS, TABLES, RunLog, dump_json, load_config
from hxai.metrics import calibration_table, paired_bootstrap, score_all
from hxai.models import (MODEL_NAMES, cart_grid, complexity, fit_model, lr_grid,
                         predict_proba, select, tune, xgb_grid)

ap = argparse.ArgumentParser()
ap.add_argument("--skip-repeated", action="store_true")
ap.add_argument("--n-jobs", type=int, default=2)
args = ap.parse_args()

cfg = load_config()
seed = cfg["seed"]
metric = cfg["selection"]["metric"]
log = RunLog("02_predictive")
b = joblib.load(MODELS / "data_bundle.joblib")
X, pop, sp, fmap = b["X"], b["pop"], b["splits"], b["fmap"]
y = pop.y
ctx = {"cfg": cfg, "numeric_cols": list(fmap.loc[fmap.kind == "numeric", "column"])}
grids = {"l1_logreg": lr_grid(cfg), "cart": cart_grid(cfg), "xgboost": xgb_grid(cfg, seed)}
log.note("grid sizes: " + ", ".join(f"{k}={len(v)}" for k, v in grids.items()))


def select_and_refit(train_idx, folds, tag):
    """Tune all models on `folds` (subsets of train_idx) and refit on train_idx."""
    out = {}
    for name in MODEL_NAMES:
        cv = tune(name, grids[name], folds, X, y, ctx, seed, n_jobs=args.n_jobs)
        best, params, extra = select(cv, grids[name], metric)
        model, info = fit_model(name, params, X.iloc[train_idx], y[train_idx], ctx, seed,
                                n_estimators=extra.get("n_estimators"))
        out[name] = {"cv": cv, "best_config": best, "params": params, "extra": extra,
                     "model": model, "fit_info": info}
        log.note(f"[{tag}] {name}: config {best} {params} {extra} "
                 f"CV {metric}={cv.loc[cv.config == best, metric].mean():.4f}")
    return out


# ---------------------------------------------------------------- main selection on dev CV
dev, test = sp["dev"], sp["test"]
main = select_and_refit(dev, sp["cv"], "main")

cv_summary = []
for name, r in main.items():
    cv = r["cv"].copy()
    cv["model"] = name
    cv.to_csv(TABLES / f"cv_results_{name}.csv", index=False)
    agg = cv.groupby("config")[["auc", "brier", "logloss"]].agg(["mean", "std"])
    best = r["best_config"]
    cv_summary.append({"model": name, "best_config": best, "n_configs": int(cv.config.nunique()),
                       **{f"cv_{m}_{s}": float(agg.loc[best, (m, s)]) for m in ("auc", "brier", "logloss")
                          for s in ("mean", "std")},
                       "params": r["params"], **r["extra"], **{f"fit_{k}": v for k, v in r["fit_info"].items()}})
    if name == "l1_logreg":
        conv = cv.groupby("config").converged.all()
        log.note(f"l1_logreg: configs with any non-converged fold: {int((~conv).sum())}")

# CART depth profile (descriptive; selection is not changed by it).
cart_cv = main["cart"]["cv"]
depth_profile = (cart_cv.groupby(["param_max_depth", "config"])[["auc", "brier"]].mean()
                 .groupby("param_max_depth").agg({"auc": "max", "brier": "min"}).reset_index())
depth_profile.to_csv(TABLES / "cart_depth_profile_cv.csv", index=False)

# ---------------------------------------------------------------- final test (single use)
Xte, yte = X.iloc[test], y[test]
preds = {name: predict_proba(r["model"], Xte) for name, r in main.items()}
test_metrics, calib, comp = {}, {}, {}
for name, p in preds.items():
    test_metrics[name] = score_all(yte, p)
    calib[name] = calibration_table(yte, p, cfg["evaluation"]["calibration_bins"])
    comp[name] = complexity(name, main[name]["model"], list(X.columns), fmap, X_eval=Xte)
boot = paired_bootstrap(yte, preds, cfg["evaluation"]["bootstrap_resamples"], seed)

thr = cfg["evaluation"]["decision_threshold"]
lab = {m: (p >= thr).astype(int) for m, p in preds.items()}
agree = {f"{a}|{b_}": float((lab[a] == lab[b_]).mean())
         for i, a in enumerate(MODEL_NAMES) for b_ in MODEL_NAMES[i + 1:]}
absdiff = {f"{a}|{b_}": {"mean": float(np.abs(preds[a] - preds[b_]).mean()),
                         "p90": float(np.quantile(np.abs(preds[a] - preds[b_]), 0.9)),
                         "share_gt_0.10": float((np.abs(preds[a] - preds[b_]) > 0.10).mean())}
           for i, a in enumerate(MODEL_NAMES) for b_ in MODEL_NAMES[i + 1:]}

pred_df = pd.DataFrame({"pos": test, "row_id": pop.row_id[test], "y": yte, **{f"p_{m}": p for m, p in preds.items()}})
pred_df.to_csv(TABLES / "test_predictions.csv", index=False)

for name, r in main.items():
    joblib.dump(r["model"], MODELS / f"final_{name}.joblib")

dump_json({"selection_metric": metric, "cv_summary": cv_summary, "test": test_metrics,
           "bootstrap": boot, "calibration": calib, "complexity": comp,
           "decision_agreement_at_0.5": agree, "abs_prob_difference": absdiff,
           "n_test": int(len(test))}, RESULTS / "predictive.json")
log.note("test: " + "; ".join(f"{m} AUC={v['auc']:.4f} Brier={v['brier']:.4f}" for m, v in test_metrics.items()))

# ---------------------------------------------------------------- repeated split study (dev only)
if not args.skip_repeated:
    rows = []
    for i, rep in enumerate(sp["repeated"]):
        res = select_and_refit(rep["train"], rep["inner"], f"rep{i}")
        ho = rep["holdout"]
        for name, r in res.items():
            p = predict_proba(r["model"], X.iloc[ho])
            rows.append({"repeat": i, "model": name, **score_all(y[ho], p),
                         "params": r["params"], **r["extra"]})
    rep_df = pd.DataFrame(rows)
    rep_df.to_csv(TABLES / "repeated_splits.csv", index=False)
    wide = rep_df.pivot(index="repeat", columns="model", values=["auc", "brier"])
    diffs = {}
    for m in ("auc", "brier"):
        for a, b_ in (("xgboost", "l1_logreg"), ("xgboost", "cart"), ("l1_logreg", "cart")):
            d = wide[(m, a)] - wide[(m, b_)]
            diffs[f"{a}-{b_}_{m}"] = {"mean": float(d.mean()), "min": float(d.min()), "max": float(d.max()),
                                      "n_positive": int((d > 0).sum()), "values": d.round(5).tolist()}
    summary = {m: {k: {"mean": float(rep_df.loc[rep_df.model == m, k].mean()),
                       "sd": float(rep_df.loc[rep_df.model == m, k].std()),
                       "min": float(rep_df.loc[rep_df.model == m, k].min()),
                       "max": float(rep_df.loc[rep_df.model == m, k].max())}
                   for k in ("auc", "brier", "logloss")} for m in MODEL_NAMES}
    dump_json({"per_model": summary, "paired_differences": diffs,
               "n_repeats": len(sp["repeated"])}, RESULTS / "repeated_splits.json")
    log.note("repeated: " + "; ".join(f"{k} mean={v['mean']:.4f}" for k, v in diffs.items()))

log.save()
