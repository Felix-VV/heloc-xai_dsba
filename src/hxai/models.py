"""Model builders, hyperparameter search and structural complexity."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .common import expand_grid_spec
from .metrics import score_all

MODEL_NAMES = ("l1_logreg", "cart", "xgboost")
LABELS = {"l1_logreg": "L1 logistic regression", "cart": "Shallow CART", "xgboost": "XGBoost"}


# ---------------------------------------------------------------- builders

def make_l1_logreg(C: float, numeric_cols: list[str], all_cols: list[str], cfg: dict, seed: int):
    mcfg = cfg["models"]["l1_logreg"]
    other = [c for c in all_cols if c not in numeric_cols]
    # Median imputation only touches numeric columns whose special codes were set to NaN;
    # the special-state indicators keep that information. Everything is standardised so the
    # L1 penalty treats columns on a common scale.
    pre = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), numeric_cols), ("rest", "passthrough", other)],
        verbose_feature_names_out=False,
    )
    clf = LogisticRegression(C=C, l1_ratio=1.0, solver=mcfg["solver"], max_iter=mcfg["max_iter"],
                             tol=mcfg["tol"], random_state=seed)
    return Pipeline([("impute", pre), ("scale", StandardScaler()), ("clf", clf)])


def make_cart(params: dict, seed: int):
    return DecisionTreeClassifier(random_state=seed, **params)


def make_xgb(params: dict, n_estimators: int, seed: int, early_stopping_rounds=None):
    return XGBClassifier(
        n_estimators=n_estimators, objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", n_jobs=1, random_state=seed,
        early_stopping_rounds=early_stopping_rounds, **params,
    )


# ---------------------------------------------------------------- search spaces

def lr_grid(cfg):
    return [{"C": float(c)} for c in expand_grid_spec(cfg["models"]["l1_logreg"]["C_grid"])]


def cart_grid(cfg):
    c = cfg["models"]["cart"]
    return [{"max_depth": int(d), "min_samples_leaf": int(m), "ccp_alpha": float(a)}
            for d in c["max_depth"] for m in c["min_samples_leaf"] for a in c["ccp_alpha"]]


def xgb_grid(cfg, seed):
    space = cfg["models"]["xgboost"]["space"]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(cfg["models"]["xgboost"]["n_random_configs"]):
        p = {}
        for k, s in space.items():
            (kind, (lo, hi)), = s.items()
            if kind == "int":
                p[k] = int(rng.integers(lo, hi + 1))
            elif kind == "uniform":
                p[k] = float(rng.uniform(lo, hi))
            elif kind == "loguniform":
                p[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        out.append(p)
    return out


# ---------------------------------------------------------------- fitting

def fit_model(name, params, X_tr, y_tr, ctx, seed, n_estimators=None):
    """Fit one model on a training part. For XGBoost without a fixed n_estimators,
    early stopping uses a stratified slice of the *training part* only.
    Returns (model, info)."""
    info = {}
    if name == "l1_logreg":
        m = make_l1_logreg(params["C"], ctx["numeric_cols"], list(X_tr.columns), ctx["cfg"], seed)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            m.fit(X_tr, y_tr)
        info["converged"] = not any(issubclass(x.category, ConvergenceWarning) for x in w)
        info["n_iter"] = int(np.max(m.named_steps["clf"].n_iter_))
        return m, info
    if name == "cart":
        return make_cart(params, seed).fit(X_tr, y_tr), info
    if name == "xgboost":
        xc = ctx["cfg"]["models"]["xgboost"]
        if n_estimators is None:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=xc["early_stopping_fraction"], random_state=seed)
            a, b = next(sss.split(np.zeros(len(y_tr)), y_tr))
            m = make_xgb(params, xc["max_estimators"], seed, xc["early_stopping_rounds"])
            m.fit(X_tr.iloc[a], y_tr[a], eval_set=[(X_tr.iloc[b], y_tr[b])], verbose=False)
            info["best_iteration"] = int(m.best_iteration)
            info["n_trees"] = int(m.best_iteration) + 1
        else:
            m = make_xgb(params, int(n_estimators), seed).fit(X_tr, y_tr, verbose=False)
            info["n_trees"] = int(n_estimators)
        return m, info
    raise ValueError(name)


def predict_proba(model, X):
    if isinstance(model, XGBClassifier) and getattr(model, "best_iteration", None) is not None \
            and model.early_stopping_rounds is not None:
        return model.predict_proba(X, iteration_range=(0, model.best_iteration + 1))[:, 1]
    return model.predict_proba(X)[:, 1]


def _eval_config(name, params, folds, X, y, ctx, seed):
    rows = []
    for k, (tr, va) in enumerate(folds):
        m, info = fit_model(name, params, X.iloc[tr], y[tr], ctx, seed + k)
        p = predict_proba(m, X.iloc[va])
        rows.append({"fold": k, **score_all(y[va], p), **info})
    return rows


def tune(name, grid, folds, X, y, ctx, seed, n_jobs=2):
    """Evaluate every configuration on the same folds; folds hold positional indices into X/y."""
    res = Parallel(n_jobs=n_jobs)(
        delayed(_eval_config)(name, p, folds, X, y, ctx, seed) for p in grid
    )
    recs = []
    for i, (p, rows) in enumerate(zip(grid, res)):
        for r in rows:
            recs.append({"config": i, **{f"param_{k}": v for k, v in p.items()}, **r})
    return pd.DataFrame(recs)


def select(cv: pd.DataFrame, grid, metric: str):
    agg = cv.groupby("config")[metric].mean()
    best = int(agg.idxmin())
    params = grid[best]
    extra = {}
    if "n_trees" in cv.columns and cv["n_trees"].notna().any():
        extra["n_estimators"] = int(np.median(cv.loc[cv.config == best, "n_trees"]))
    return best, params, extra


# ---------------------------------------------------------------- complexity

def complexity(name, model, X_cols, feature_map, X_eval=None) -> dict:
    var_of = dict(zip(feature_map.column, feature_map.variable))
    if name == "l1_logreg":
        coef = model.named_steps["clf"].coef_.ravel()
        # The ColumnTransformer reorders columns (numeric first), so pair coefficients
        # with the pipeline's output names, not with the input column order.
        transformed_cols = model[:-1].get_feature_names_out()
        nz = [c for c, b in zip(transformed_cols, coef) if b != 0]
        return {"nonzero_coefficients": len(nz), "input_columns": len(X_cols),
                "original_variables_used": len({var_of[c] for c in nz})}
    if name == "cart":
        t = model.tree_
        used = {X_cols[f] for f in t.feature if f >= 0}
        out = {"depth": int(model.get_depth()), "leaves": int(model.get_n_leaves()),
               "internal_nodes": int(t.node_count - model.get_n_leaves()),
               "original_variables_used": len({var_of[c] for c in used})}
        if X_eval is not None:
            lengths = np.asarray(model.decision_path(X_eval).sum(axis=1)).ravel() - 1
            out["mean_path_length"] = float(lengths.mean())
            out["max_path_length"] = int(lengths.max())
        return out
    if name == "xgboost":
        df = model.get_booster().trees_to_dataframe()
        leaves = df[df.Feature == "Leaf"]
        split_feats = set(df.loc[df.Feature != "Leaf", "Feature"])
        return {"trees": int(df.Tree.nunique()), "total_leaves": int(len(leaves)),
                "internal_nodes": int((df.Feature != "Leaf").sum()),
                "max_depth_param": int(model.get_params()["max_depth"]),
                "original_variables_used": len({var_of[c] for c in split_feats})}
    raise ValueError(name)
