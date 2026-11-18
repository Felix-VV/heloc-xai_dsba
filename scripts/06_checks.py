"""Checks for errors that would change the conclusions. Exits non-zero on failure."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

from hxai.common import LOGS, MODELS, RESULTS, TABLES, load_config, load_json
from hxai.data import load_raw, make_population, make_splits
from hxai.explain import lr_attributions, margin, tree_shap
from hxai.metrics import score_all
from hxai.models import predict_proba

cfg = load_config()
b = joblib.load(MODELS / "data_bundle.joblib")
X, pop, sp, fmap = b["X"], b["pop"], b["splits"], b["fmap"]
models = {m: joblib.load(MODELS / f"final_{m}.joblib") for m in ("l1_logreg", "cart", "xgboost")}
pj = load_json(RESULTS / "predictive.json")
pred = pd.read_csv(TABLES / "test_predictions.csv")
dev, test = sp["dev"], sp["test"]
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


# 1. Splits are reproducible from the config and disjoint; tuning folds live inside dev.
raw = load_raw(cfg)
sp2 = make_splits(make_population(raw, cfg), cfg)
check("splits reproducible from config", np.array_equal(sp2["test"], test) and np.array_equal(sp2["dev"], dev))
check("dev/test disjoint and exhaustive", not set(dev) & set(test) and len(dev) + len(test) == len(pop.y))
check("CV folds use development rows only", all(set(tr) | set(va) <= set(dev) for tr, va in sp["cv"]))
check("repeated-split rows are development rows only",
      all(set(r["train"]) | set(r["holdout"]) <= set(dev) for r in sp["repeated"]))

# 2. Label direction: y = 1 is "Bad".
lab = raw.loc[pop.row_id, cfg["data"]["target"]].to_numpy()
check("y == 1 exactly when RiskPerformance == 'Bad'", np.array_equal(pop.y == 1, lab == "Bad"))
rho = spearmanr(X.ExternalRiskEstimate, pop.y, nan_policy="omit").statistic
check("ExternalRiskEstimate negatively associated with Bad", rho < 0, f"rho={rho:.3f}")

# 3. Preprocessing statistics come from the development set only.
lr = models["l1_logreg"]
imp = lr.named_steps["impute"].named_transformers_["num"]
num_cols = list(fmap.loc[fmap.kind == "numeric", "column"])
dev_medians = X.iloc[dev][num_cols].median().to_numpy()
all_medians = X[num_cols].median().to_numpy()
check("LR imputation medians equal development-set medians", np.allclose(imp.statistics_, dev_medians))
check("  (development medians differ from full-population medians somewhere)",
      not np.allclose(dev_medians, all_medians), "confirms the check above is informative")
Z = lr[:-1].transform(X.iloc[dev])
check("LR scaler fitted on development rows (transformed dev mean ≈ 0)", np.allclose(Z.mean(0), 0, atol=1e-8))

# 4. Saved predictions align with positions/rows and are reproducible from saved models.
check("prediction rows are the test rows", np.array_equal(pred.pos.to_numpy(), test)
      and np.array_equal(pred.row_id.to_numpy(), pop.row_id[test]) and np.array_equal(pred.y.to_numpy(), pop.y[test]))
for m, model in models.items():
    p = predict_proba(model, X.iloc[test])
    check(f"{m}: saved test predictions reproducible", np.allclose(p, pred[f"p_{m}"], atol=1e-6))
    s = score_all(pred.y, pred[f"p_{m}"])
    check(f"{m}: reported test AUC/Brier recomputed on the same rows",
          np.isclose(s["auc"], pj["test"][m]["auc"]) and np.isclose(s["brier"], pj["test"][m]["brier"]))

# 5. Explanations refer to the right model, class and scale.
bg = X.iloc[np.load(MODELS / "background_positions.npy")]
xs = X.iloc[test[:200]]
base, sv = tree_shap(models["xgboost"], xs, bg)
mg = margin(models["xgboost"], xs)
check("TreeSHAP additivity on log-odds (base + sum = margin)", np.allclose(base + sv.sum(axis=1), mg, atol=1e-4))
check("XGB margin is the logit of P(Bad)", np.allclose(expit(mg), pred.p_xgboost.iloc[:200], atol=1e-6))
check("XGB base value = mean margin over background", np.isclose(base, margin(models["xgboost"], bg).mean(), atol=1e-4))
lb, la = lr_attributions(lr, xs, bg)
check("LR attributions sum to LR logit", np.allclose(lb + la.sum(axis=1), lr.decision_function(xs)))
check("LR logit is the logit of P(Bad)", np.allclose(expit(lr.decision_function(xs)), pred.p_l1_logreg.iloc[:200]))
check("CART classes ordered (0=Good, 1=Bad)", list(models["cart"].classes_) == [0, 1]
      and list(models["xgboost"].classes_) == [0, 1] and list(lr.classes_) == [0, 1])

# 6. Structural complexity pairs coefficients with the right columns.
coef_tab = pd.read_csv(TABLES / "lr_coefficients.csv")
vars_from_table = coef_tab.loc[coef_tab.coef_standardized != 0, "variable"].nunique()
cx = pj["complexity"]["l1_logreg"]
check("LR variable count equals variables with non-zero rows in the coefficient table",
      cx["original_variables_used"] == vars_from_table
      and cx["nonzero_coefficients"] == int((coef_tab.coef_standardized != 0).sum()),
      f"{cx['original_variables_used']} variables, {cx['nonzero_coefficients']} coefficients")

# 7. Report numbers are generated, not typed.
macros = (RESULTS / "report_numbers.tex").read_text()
check("report macros file present and generated from JSON", "Auto-generated" in macros
      and f"\\newcommand{{\\TestAUCLR}}{{{pj['test']['l1_logreg']['auc']:.3f}}}" in macros)

n_fail = sum(not ok for _, ok, _ in results)
with open(LOGS / "06_checks.txt", "w") as f:
    for name, ok, d in results:
        f.write(("PASS " if ok else "FAIL ") + name + (f"  [{d}]" if d else "") + "\n")
    f.write(f"\n{len(results) - n_fail}/{len(results)} checks passed\n")
print(f"{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
