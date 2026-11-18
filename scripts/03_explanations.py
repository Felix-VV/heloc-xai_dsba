"""Same-case explanation comparison under a pre-specified case-selection rule."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import export_text

from hxai.common import MODELS, RESULTS, TABLES, RunLog, dump_json, load_config
from hxai.explain import (cart_path, group_by_variable, lr_attributions, lr_coef_table, margin,
                          tree_shap)

cfg = load_config()
seed = cfg["seed"]
log = RunLog("03_explanations")
b = joblib.load(MODELS / "data_bundle.joblib")
X, pop, sp, fmap = b["X"], b["pop"], b["splits"], b["fmap"]
models = {m: joblib.load(MODELS / f"final_{m}.joblib") for m in ("l1_logreg", "cart", "xgboost")}
pred = pd.read_csv(TABLES / "test_predictions.csv")

# Fixed background: random rows of the development set (reference population = dev set).
rng = np.random.default_rng(seed)
bg_pos = np.sort(rng.choice(sp["dev"], size=cfg["explanation"]["background_size"], replace=False))
np.save(MODELS / "background_positions.npy", bg_pos)
X_bg = X.iloc[bg_pos]

# ---------------------------------------------------------------- case selection (rules fixed in config)
rules = cfg["explanation"]["case_rules"]
P = pred[["p_l1_logreg", "p_cart", "p_xgboost"]].to_numpy()
pbar, spread = P.mean(1), P.max(1) - P.min(1)
n_special = X.iloc[pred.pos][fmap.loc[fmap.kind == "special", "column"]].sum(axis=1).to_numpy()
thr = rules["near_threshold"]

orders = {
    "low_risk": np.argsort(np.abs(pbar - np.quantile(pbar, rules["low_risk_quantile"])), kind="stable"),
    "high_risk": np.argsort(np.abs(pbar - np.quantile(pbar, rules["high_risk_quantile"])), kind="stable"),
    "near_threshold": np.argsort(np.abs(P - thr).max(1), kind="stable"),
    "disagreement": np.argsort(-spread, kind="stable"),
    "special_codes": np.lexsort((np.arange(len(pbar)), -n_special)),
}
chosen = {}
for name, order in orders.items():
    for i in order:
        if i in chosen.values():
            continue
        if name == "special_codes" and n_special[i] < rules["min_special_states"]:
            raise RuntimeError("no case meets the special-state rule")
        chosen[name] = int(i)
        break

case_rows, long_rows, paths = [], [], {}
var_order = list(dict.fromkeys(fmap.variable))
for label, i in chosen.items():
    pos = int(pred.pos.iloc[i])
    x = X.iloc[[pos]]
    raw = pop.X_raw.iloc[pos]
    # Logistic regression: exact additive decomposition relative to the background mean.
    lr_base, lr_c = lr_attributions(models["l1_logreg"], x, X_bg)
    lr_logit = float(models["l1_logreg"].decision_function(x)[0])
    assert abs(lr_base + lr_c.sum(axis=1).iloc[0] - lr_logit) < 1e-8
    # XGBoost: interventional TreeSHAP on log-odds; computed twice to confirm determinism.
    xb_base, xb_s = tree_shap(models["xgboost"], x, X_bg)
    xb_base2, xb_s2 = tree_shap(models["xgboost"], x, X_bg)
    assert xb_base == xb_base2 and np.array_equal(xb_s.values, xb_s2.values)
    xb_margin = float(margin(models["xgboost"], x)[0])
    add_err = abs(xb_base + float(xb_s.sum(axis=1).iloc[0]) - xb_margin)
    # Interventional base value must equal the mean model output over the background.
    assert abs(xb_base - float(margin(models["xgboost"], X_bg).mean())) < 1e-4
    assert add_err < 1e-4, add_err
    path = cart_path(models["cart"], x.iloc[0], list(X.columns))
    paths[label] = path
    case_rows.append({
        "case": label, "pos": pos, "row_id": int(pop.row_id[pos]), "y": int(pop.y[pos]),
        "p_l1_logreg": float(pred.p_l1_logreg.iloc[i]), "p_cart": float(pred.p_cart.iloc[i]),
        "p_xgboost": float(pred.p_xgboost.iloc[i]), "n_special_states": int(n_special[i]),
        "lr_base_logit": lr_base, "lr_logit": lr_logit, "xgb_base_logit": xb_base,
        "xgb_logit": xb_margin, "xgb_additivity_error": add_err,
        "cart_leaf": path["leaf"], "cart_leaf_train_n": path["leaf_train_samples"],
        "cart_path_length": len(path["steps"]),
    })
    g_lr = group_by_variable(lr_c, fmap).iloc[0]
    g_xb = group_by_variable(xb_s, fmap).iloc[0]
    cart_vars = {fmap.set_index("column").variable[s["column"]] for s in path["steps"]}
    for v in var_order:
        long_rows.append({"case": label, "variable": v, "raw_value": int(raw[v]),
                          "lr_contribution": float(g_lr[v]), "xgb_shap_sum": float(g_xb[v]),
                          "on_cart_path": v in cart_vars})

cases = pd.DataFrame(case_rows)
cases.to_csv(TABLES / "explanation_cases.csv", index=False)
pd.DataFrame(long_rows).to_csv(TABLES / "explanation_cases_by_variable.csv", index=False)
dump_json({"rules": rules, "chosen_test_index": chosen, "cart_paths": paths,
           "background_size": int(len(bg_pos))}, RESULTS / "explanations.json")

coef = lr_coef_table(models["l1_logreg"], fmap)
coef.to_csv(TABLES / "lr_coefficients.csv", index=False)
with open(TABLES / "cart_tree.txt", "w") as f:
    f.write(export_text(models["cart"], feature_names=list(X.columns), show_weights=False, decimals=1))

log.note("cases: " + ", ".join(f"{k}=test#{v}" for k, v in chosen.items()))
log.note(f"max XGB additivity error {cases.xgb_additivity_error.max():.2e}")
log.save()
