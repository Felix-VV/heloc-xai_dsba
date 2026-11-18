"""Exploratory upstream-score ablation with unchanged customers, splits, grids and selection."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
import numpy as np
import pandas as pd
from hxai.common import ROOT, MODELS, RESULTS, TABLES, RunLog, dump_json, load_config, load_json
from hxai.data import design_matrix
from hxai.explain import lr_coef_table
from hxai.metrics import calibration_table, paired_bootstrap, score_all
from hxai.models import MODEL_NAMES, cart_grid, complexity, fit_model, lr_grid, predict_proba, select, tune, xgb_grid

ap = argparse.ArgumentParser()
ap.add_argument("--n-jobs", type=int, default=2)
ap.add_argument("--resume", action="store_true", help="Reuse completed repeat checkpoints with an identical code/config signature.")
args = ap.parse_args()
cfg = load_config()
seed = cfg["seed"]
log = RunLog("07_ablation")
b = joblib.load(MODELS / "data_bundle.joblib")
pop, sp = b["pop"], b["splits"]
y = pop.y
excluded = "ExternalRiskEstimate"
X, fmap = design_matrix(pop.X_raw.drop(columns=[excluded]), cfg)
assert excluded not in set(fmap.variable)
assert len(X) == len(b["X"])
fmap.to_csv(TABLES / "ablation_feature_map.csv", index=False)
ctx = {"cfg": cfg, "numeric_cols": list(fmap.loc[fmap.kind == "numeric", "column"])}
grids = {"l1_logreg": lr_grid(cfg), "cart": cart_grid(cfg), "xgboost": xgb_grid(cfg, seed)}
signature_payload = {
    "config": cfg, "excluded": excluded, "versions": log.record["versions"],
    "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), ROOT / "src/hxai/data.py", ROOT / "src/hxai/models.py", ROOT / "src/hxai/metrics.py"]},
}
signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
checkpoints = RESULTS / "ablation_checkpoints"
checkpoints.mkdir(exist_ok=True)

def select_and_refit(train_idx, folds, tag, save_cv=False):
    result = {}
    for name in MODEL_NAMES:
        cv = tune(name, grids[name], folds, X, y, ctx, seed, n_jobs=args.n_jobs)
        best, params, extra = select(cv, grids[name], cfg["selection"]["metric"])
        model, info = fit_model(name, params, X.iloc[train_idx], y[train_idx], ctx, seed,
                                n_estimators=extra.get("n_estimators"))
        best_rows = cv.loc[cv.config == best]
        result[name] = {"model": model, "selection": {
            "model": name, "best_config": best, "params": params, **extra,
            "n_configs": len(grids[name]),
            **{f"cv_{m}_mean": float(best_rows[m].mean()) for m in ["auc", "brier", "logloss"]},
            **{f"cv_{m}_std": float(best_rows[m].std()) for m in ["auc", "brier", "logloss"]},
            "fit_info": info,
        }}
        if save_cv:
            cv.to_csv(TABLES / f"ablation_cv_results_{name}.csv", index=False)
        log.note(f"[{tag}] {name}: CV Brier={best_rows.brier.mean():.5f}, params={params} {extra}")
    return result

dev, test = sp["dev"], sp["test"]
log.note(f"No-score predictors: {len(X.columns)} encoded columns, {fmap.variable.nunique()} original variables; all customer splits unchanged.")
main = select_and_refit(dev, sp["cv"], "main", save_cv=True)
base = load_json(RESULTS / "predictive.json")
base_pred = pd.read_csv(TABLES / "test_predictions.csv")
assert np.array_equal(base_pred.row_id.to_numpy(), pop.row_id[test])
assert np.array_equal(base_pred.y.to_numpy(), y[test])

per_model = {}
pred_df = pd.DataFrame({"pos": test, "row_id": pop.row_id[test], "y": y[test]})
for name, r in main.items():
    full = base_pred[f"p_{name}"].to_numpy()
    no = predict_proba(r["model"], X.iloc[test])
    met = score_all(y[test], no)
    boot = paired_bootstrap(y[test], {"full": full, "no_score": no}, cfg["evaluation"]["bootstrap_resamples"], seed)
    changes = np.abs(full - no)
    per_model[name] = {
        "selection": r["selection"], "full": base["test"][name], "no_score": met,
        "full_minus_no_score": {m: float(base["test"][name][m] - met[m]) for m in ["auc", "brier", "logloss"]},
        "bootstrap": boot,
        "calibration": calibration_table(y[test], no, cfg["evaluation"]["calibration_bins"]),
        "complexity": complexity(name, r["model"], list(X.columns), fmap, X_eval=X.iloc[test]),
        "individual_change": {"mean_absolute": float(changes.mean()), "p90_absolute": float(np.quantile(changes, .9)),
                              "fraction_gt_0.10": float((changes > .1).mean()),
                              "label_disagreement_at_0.5": float(((full >= .5) != (no >= .5)).mean())},
    }
    pred_df[f"full_{name}"] = full
    pred_df[f"no_score_{name}"] = no
    joblib.dump(r["model"], MODELS / f"ablation_no_score_{name}.joblib")
    if name == "l1_logreg":
        lr_coef_table(r["model"], fmap).to_csv(TABLES / "ablation_lr_coefficients.csv", index=False)
    log.note(f"test {name}: full AUC={base['test'][name]['auc']:.4f}, no-score AUC={met['auc']:.4f}; no-score Brier={met['brier']:.5f}")
pred_df.to_csv(TABLES / "ablation_test_predictions.csv", index=False)

all_repeated = []
for i, rep in enumerate(sp["repeated"]):
    ck = checkpoints / f"repeat_{i:02d}.json"
    saved = load_json(ck) if args.resume and ck.exists() else None
    if saved is not None and saved.get("signature") == signature:
        rows = saved["rows"]
        log.note(f"repeat {i}: reused verified-signature checkpoint")
    else:
        fit = select_and_refit(rep["train"], rep["inner"], f"repeat{i}")
        rows = []
        for name, r in fit.items():
            p = predict_proba(r["model"], X.iloc[rep["holdout"]])
            rows.append({"repeat": i, "model": name, **score_all(y[rep["holdout"]], p),
                         "params": r["selection"]["params"],
                         "n_estimators": r["selection"].get("n_estimators")})
        dump_json({"signature": signature, "rows": rows}, ck)
    all_repeated.extend(rows)

rep_df = pd.DataFrame(all_repeated)
rep_df.to_csv(TABLES / "ablation_repeated_splits.csv", index=False)
full_rep = pd.read_csv(TABLES / "repeated_splits.csv")
joined = full_rep.merge(rep_df, on=["repeat", "model"], suffixes=("_full", "_no_score"), validate="one_to_one")
for m in ["auc", "brier", "logloss"]:
    joined[f"delta_{m}_full_minus_no_score"] = joined[f"{m}_full"] - joined[f"{m}_no_score"]
joined.to_csv(TABLES / "ablation_paired_repeated_splits.csv", index=False)
repeat_summary = {}
for name in MODEL_NAMES:
    d = joined.loc[joined.model == name]
    repeat_summary[name] = {}
    for m in ["auc", "brier", "logloss"]:
        delta = d[f"delta_{m}_full_minus_no_score"]
        repeat_summary[name][m] = {
            "full_mean": float(d[f"{m}_full"].mean()), "full_sd": float(d[f"{m}_full"].std()),
            "no_score_mean": float(d[f"{m}_no_score"].mean()), "no_score_sd": float(d[f"{m}_no_score"].std()),
            "delta_mean": float(delta.mean()), "delta_min": float(delta.min()), "delta_max": float(delta.max()),
            "delta_positive_count": int((delta > 0).sum()), "values": delta.tolist(),
        }

dump_json({
    "status": "complete", "analysis": "exploratory upstream-score ablation", "signature": signature,
    "excluded_variable": excluded, "n_population": len(y), "n_dev": len(dev), "n_test": len(test),
    "n_columns_full": len(b["X"].columns), "n_columns_no_score": len(X.columns),
    "n_variables_no_score": int(fmap.variable.nunique()),
    "test_row_ids_sha256": hashlib.sha256(pop.row_id[test].tobytes()).hexdigest(),
    "selection_metric": cfg["selection"]["metric"], "per_model": per_model,
    "repeated": {"n_repeats": len(sp["repeated"]), "per_model": repeat_summary},
    "interpretation": "Differences are full minus no-score. Test CIs condition on fitted models. The test set had been inspected; all new findings are exploratory. Removing this score is not proof of end-to-end transparency or causality.",
}, RESULTS / "ablation.json")
log.save(signature=signature)
