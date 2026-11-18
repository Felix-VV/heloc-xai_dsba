"""Stability of predictions and explanations.

A. Retraining stability: hyperparameters fixed at the selected values; each run refits
   on a 90% subsample of the development set (and, for XGBoost, a new seed). A separate
   XGBoost arm changes only the seed on the full development set. Evaluation cases and
   the explanation background are fixed across runs. No retuning, and no claim that
   these refits form a verified Rashomon set.
B. Background sensitivity: the final XGBoost (and, for comparison, logistic) model is
   fixed; backgrounds of fixed size are redrawn from the same reference population
   (development set). A separate check swaps the reference population (Good-only /
   Bad-only), which changes the question the explanation answers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from hxai.common import MODELS, RESULTS, TABLES, RunLog, dump_json, load_config, load_json
from hxai.explain import (cart_path_variables, group_by_variable, lr_attributions,
                          pairwise_jaccard, pairwise_rel_l1, pairwise_topk_overlap,
                          sign_agreement, sign_conflict, tree_shap)
from hxai.models import fit_model, predict_proba

cfg = load_config()
seed = cfg["seed"]
scfg = cfg["stability"]
log = RunLog("04_stability")
b = joblib.load(MODELS / "data_bundle.joblib")
X, pop, sp, fmap = b["X"], b["pop"], b["splits"], b["fmap"]
y = pop.y
var_of = dict(zip(fmap.column, fmap.variable))
variables = list(dict.fromkeys(fmap.variable))
ctx = {"cfg": cfg, "numeric_cols": list(fmap.loc[fmap.kind == "numeric", "column"])}
pj = load_json(RESULTS / "predictive.json")
sel = {r["model"]: (r["params"], r.get("n_estimators")) for r in pj["cv_summary"]}
final = {m: joblib.load(MODELS / f"final_{m}.joblib") for m in ("l1_logreg", "cart", "xgboost")}
bg_pos = np.load(MODELS / "background_positions.npy")
X_bg = X.iloc[bg_pos]
thr = cfg["evaluation"]["decision_threshold"]
nz = scfg["near_zero"]

# Fixed evaluation cases: stratified subset of the test set plus the showcase cases.
test = sp["test"]
sss = StratifiedShuffleSplit(n_splits=1, test_size=scfg["eval_cases"], random_state=seed)
_, e = next(sss.split(np.zeros(len(test)), y[test]))
showcase = pd.read_csv(TABLES / "explanation_cases.csv")
eval_pos = np.unique(np.concatenate([test[e], showcase.pos.to_numpy()]))
X_ev = X.iloc[eval_pos]
np.save(MODELS / "stability_eval_positions.npy", eval_pos)
log.note(f"evaluation cases: {len(eval_pos)}")


def grouped(attr):
    return group_by_variable(attr, fmap).to_numpy()


def summarise(tag, method, preds=None, attrs=None, path_sets=None):
    """Per-case stability measures for one (analysis, method) arm."""
    out = pd.DataFrame({"pos": eval_pos, "analysis": tag, "method": method})
    if preds is not None:
        Pm = np.stack(preds)                         # runs x cases
        out["pred_sd"] = Pm.std(0, ddof=1)
        out["pred_range"] = Pm.max(0) - Pm.min(0)
        lab = Pm >= thr
        out["label_not_unanimous"] = (lab.any(0) & ~lab.all(0)).astype(int)
    if attrs is not None:
        for k in scfg["top_k"]:
            out[f"top{k}_overlap"] = pairwise_topk_overlap(attrs, k)
        out["sign_agreement"] = sign_agreement(attrs, nz)
        out["sign_conflict"] = sign_conflict(attrs, nz)
        np.save(MODELS / f"attr_{tag}_{method}.npy", np.stack(attrs))
        out["rel_l1"] = pairwise_rel_l1(attrs)
        out["attr_sd_mean"] = np.stack(attrs).std(0, ddof=1).mean(1)
    if path_sets is not None:
        out["path_var_jaccard"] = pairwise_jaccard(path_sets)
    return out


per_case, extra = [], {}

# ---------------------------------------------------------------- A. retraining stability
R = scfg["retrain"]["n_runs"]
frac = scfg["retrain"]["subsample_fraction"]
dev = sp["dev"]
arms = {m: {"preds": [], "attrs": [], "paths": []} for m in ("l1_logreg", "cart", "xgboost")}
lr_coefs, cart_struct = [], []
for r in range(R):
    ss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed + 1000 + r)
    a, _ = next(ss.split(np.zeros(len(dev)), y[dev]))
    tr = dev[a]
    for m in arms:
        params, n_est = sel[m]
        model, info = fit_model(m, params, X.iloc[tr], y[tr], ctx, seed + 1000 + r, n_estimators=n_est)
        arms[m]["preds"].append(predict_proba(model, X_ev))
        if m == "l1_logreg":
            _, att = lr_attributions(model, X_ev, X_bg)
            arms[m]["attrs"].append(grouped(att))
            sc = model.named_steps["scale"].scale_
            cols = list(model[:-1].get_feature_names_out())
            lr_coefs.append(pd.Series(model.named_steps["clf"].coef_.ravel() / sc, index=cols)[list(X.columns)])
        elif m == "xgboost":
            _, att = tree_shap(model, X_ev, X_bg)
            arms[m]["attrs"].append(grouped(att))
        else:
            arms[m]["paths"].append(cart_path_variables(model, X_ev, var_of))
            t = model.tree_
            cart_struct.append({"run": r, "root_variable": var_of[X.columns[t.feature[0]]],
                                "root_column": X.columns[t.feature[0]], "leaves": int(model.get_n_leaves()),
                                "depth": int(model.get_depth())})
    log.note(f"retrain run {r + 1}/{R} done")

per_case.append(summarise("retrain", "l1_logreg", arms["l1_logreg"]["preds"], arms["l1_logreg"]["attrs"]))
per_case.append(summarise("retrain", "xgboost", arms["xgboost"]["preds"], arms["xgboost"]["attrs"]))
per_case.append(summarise("retrain", "cart", arms["cart"]["preds"], path_sets=arms["cart"]["paths"]))

C = pd.DataFrame(lr_coefs)
lr_sel = pd.DataFrame({"column": C.columns, "variable": [var_of[c] for c in C.columns],
                       "selected_frac": (C != 0).mean().to_numpy(),
                       "positive_frac_when_selected": [((C[c] > 0).sum() / max((C[c] != 0).sum(), 1)) for c in C.columns],
                       "coef_per_unit_median": C.median().to_numpy()})
lr_sel.to_csv(TABLES / "stability_lr_selection.csv", index=False)
final_nz = set(np.array(final["l1_logreg"][:-1].get_feature_names_out())[final["l1_logreg"].named_steps["clf"].coef_.ravel() != 0])
nnz = (C != 0).sum(axis=1)
extra["lr_retrain"] = {
    "nonzero_count_min": int(nnz.min()), "nonzero_count_max": int(nnz.max()),
    "columns_always_selected": int((lr_sel.selected_frac == 1).sum()),
    "columns_never_selected": int((lr_sel.selected_frac == 0).sum()),
    "columns_sometimes_selected": int(((lr_sel.selected_frac > 0) & (lr_sel.selected_frac < 1)).sum()),
    "sign_flips_columns": int(((lr_sel.positive_frac_when_selected > 0) & (lr_sel.positive_frac_when_selected < 1)).sum()),
    "final_model_nonzero": len(final_nz),
}
cs = pd.DataFrame(cart_struct)
cs.to_csv(TABLES / "stability_cart_structure.csv", index=False)
extra["cart_retrain"] = {"root_variable_counts": cs.root_variable.value_counts().to_dict(),
                         "leaves_min": int(cs.leaves.min()), "leaves_max": int(cs.leaves.max())}

# Seed-only arm (XGBoost on the full development set).
params, n_est = sel["xgboost"]
sp_preds, sp_attrs = [], []
for r in range(R):
    model, _ = fit_model("xgboost", params, X.iloc[dev], y[dev], ctx, seed + 2000 + r, n_estimators=n_est)
    sp_preds.append(predict_proba(model, X_ev))
    _, att = tree_shap(model, X_ev, X_bg)
    sp_attrs.append(grouped(att))
per_case.append(summarise("seed_only", "xgboost", sp_preds, sp_attrs))
log.note("seed-only arm done")

# ---------------------------------------------------------------- B. background sensitivity
rng = np.random.default_rng(seed + 3000)
bg_summary = {}
for size in scfg["background"]["sizes"]:
    xb_attr, lr_attr, xb_base, lr_base = [], [], [], []
    for d in range(scfg["background"]["n_draws"]):
        pos = rng.choice(dev, size=size, replace=False)
        bb, att = tree_shap(final["xgboost"], X_ev, X.iloc[pos])
        xb_attr.append(grouped(att)); xb_base.append(bb)
        lb, latt = lr_attributions(final["l1_logreg"], X_ev, X.iloc[pos])
        lr_attr.append(grouped(latt)); lr_base.append(lb)
    per_case.append(summarise(f"background_{size}", "xgboost", attrs=xb_attr))
    per_case.append(summarise(f"background_{size}", "l1_logreg", attrs=lr_attr))
    bg_summary[size] = {"xgb_base_sd": float(np.std(xb_base, ddof=1)), "xgb_base_range": float(np.ptp(xb_base)),
                        "lr_base_sd": float(np.std(lr_base, ddof=1)), "lr_base_range": float(np.ptp(lr_base))}
    log.note(f"background size {size} done")
extra["background_base_values"] = bg_summary

# Reference-population swap for the showcase cases (changes the explanation question).
X_sc = X.iloc[showcase.pos.to_numpy()]
swap_rows = []
dev_good, dev_bad = dev[y[dev] == 0], dev[y[dev] == 1]
rs = np.random.default_rng(seed + 4000)
refs = {"dev_all": X_bg,
        "dev_good_only": X.iloc[rs.choice(dev_good, cfg["explanation"]["background_size"], replace=False)],
        "dev_bad_only": X.iloc[rs.choice(dev_bad, cfg["explanation"]["background_size"], replace=False)]}
for ref_name, Xr in refs.items():
    bb, att = tree_shap(final["xgboost"], X_sc, Xr)
    g = group_by_variable(att, fmap)
    for j, case in enumerate(showcase.case):
        top = g.iloc[j].abs().sort_values(ascending=False, kind="stable").index[:3].tolist()
        swap_rows.append({"case": case, "reference": ref_name, "base_logit": bb,
                          "sum_attr": float(g.iloc[j].sum()), "top3": "|".join(top),
                          **{f"attr_{v}": float(g.iloc[j][v]) for v in variables}})
pd.DataFrame(swap_rows).to_csv(TABLES / "stability_reference_swap.csv", index=False)

# ---------------------------------------------------------------- save
pc = pd.concat(per_case, ignore_index=True)
pc.to_csv(TABLES / "stability_per_case.csv", index=False)
metrics = ["pred_sd", "pred_range", "label_not_unanimous", "top3_overlap", "top5_overlap",
           "sign_agreement", "sign_conflict", "rel_l1", "attr_sd_mean", "path_var_jaccard"]
summ = {}
for (an, me), g in pc.groupby(["analysis", "method"]):
    summ[f"{an}|{me}"] = {k: {"mean": float(g[k].mean()), "median": float(g[k].median()),
                              "p10": float(g[k].quantile(0.1)), "p90": float(g[k].quantile(0.9))}
                          for k in metrics if k in g and g[k].notna().any()}
dump_json({"n_eval_cases": int(len(eval_pos)), "n_runs": R, "summary": summ, **extra},
          RESULTS / "stability.json")
log.save()
