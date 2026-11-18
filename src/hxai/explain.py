"""Model-specific explanations on a common log-odds scale, plus stability measures.

Conventions
- All attributions are on the log-odds (logit) scale of P(Bad), never probability.
- Attributions are relative to a reference (background) distribution: the
  contribution of feature j is the change in log-odds attributed to j when moving
  from the background to the case. For the logistic regression this is computed
  exactly as w_j * (T(x)_j - mean_bg T(X)_j), which is what interventional SHAP
  returns for a linear model with the same background.
- Grouping by original variable sums column attributions (numeric part +
  special-state indicators + dummies). For XGBoost this is a sum of SHAP values,
  not a Shapley value of the group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


# ---------------------------------------------------------------- logistic regression

def lr_transform(pipe, X):
    return pipe[:-1].transform(X)


def lr_attributions(pipe, X: pd.DataFrame, X_ref: pd.DataFrame):
    clf = pipe.named_steps["clf"]
    w = clf.coef_.ravel()
    b = float(clf.intercept_[0])
    cols = list(pipe[:-1].get_feature_names_out())
    Z = lr_transform(pipe, X)
    Zr = lr_transform(pipe, X_ref).mean(axis=0)
    contrib = (Z - Zr) * w
    base = b + float(Zr @ w)
    out = pd.DataFrame(contrib, columns=cols, index=X.index)[list(X.columns)]
    return base, out


def lr_coef_table(pipe, feature_map: pd.DataFrame) -> pd.DataFrame:
    clf, scaler = pipe.named_steps["clf"], pipe.named_steps["scale"]
    cols = list(pipe[:-1].get_feature_names_out())
    w = clf.coef_.ravel()
    sd = scaler.scale_
    df = pd.DataFrame({"column": cols, "coef_standardized": w, "column_sd": sd,
                       "coef_per_unit": w / sd})
    df["odds_ratio_per_unit"] = np.exp(df.coef_per_unit)
    df = df.merge(feature_map, on="column", how="left")
    return df


# ---------------------------------------------------------------- CART

def cart_path(tree, x: pd.Series, cols: list[str]) -> dict:
    t = tree.tree_
    node, steps = 0, []
    while t.children_left[node] != -1:
        f, thr = t.feature[node], t.threshold[node]
        val = x.iloc[f]
        if np.isnan(val):
            go_left = bool(t.missing_go_to_left[node])
            how = "missing"
        else:
            go_left = val <= thr
            how = "value"
        steps.append({"node": int(node), "column": cols[f], "threshold": float(thr),
                      "value": None if np.isnan(val) else float(val), "routed": how,
                      "direction": "<=" if go_left else ">",
                      "missing_go_left": bool(t.missing_go_to_left[node])})
        node = t.children_left[node] if go_left else t.children_right[node]
    leaf_proba = tree.predict_proba(x.to_frame().T)[0, 1]
    return {"steps": steps, "leaf": int(node), "leaf_train_samples": int(t.n_node_samples[node]),
            "leaf_p_bad": float(leaf_proba)}


def cart_path_variables(tree, X: pd.DataFrame, var_of: dict) -> list[frozenset]:
    """Set of original variables tested on each row's decision path."""
    dp = tree.decision_path(X)
    feat = tree.tree_.feature
    cols = list(X.columns)
    out = []
    for i in range(X.shape[0]):
        nodes = dp.indices[dp.indptr[i]:dp.indptr[i + 1]]
        out.append(frozenset(var_of[cols[feat[n]]] for n in nodes if feat[n] >= 0))
    return out


# ---------------------------------------------------------------- XGBoost / TreeSHAP

def tree_shap(model, X: pd.DataFrame, background: pd.DataFrame, check_additivity=True):
    """Interventional TreeSHAP on the raw (log-odds) output. Deterministic given inputs."""
    # An explicit masker: by default shap silently subsamples backgrounds above 100 rows.
    masker = shap.maskers.Independent(background, max_samples=len(background))
    expl = shap.TreeExplainer(model, data=masker, feature_perturbation="interventional",
                              model_output="raw")
    if expl.data.shape[0] != len(background):
        raise RuntimeError("background was subsampled")
    vals = expl.shap_values(X, check_additivity=check_additivity)
    base = float(np.ravel(expl.expected_value)[0])
    return base, pd.DataFrame(vals, columns=X.columns, index=X.index)


def margin(model, X):
    return model.predict(X, output_margin=True)


# ---------------------------------------------------------------- grouping & stability

def group_by_variable(attr: pd.DataFrame, feature_map: pd.DataFrame) -> pd.DataFrame:
    var_of = dict(zip(feature_map.column, feature_map.variable))
    order = list(dict.fromkeys(feature_map.variable))
    g = attr.T.groupby(attr.columns.map(var_of)).sum().T
    return g[order]


def topk_sets(A: np.ndarray, k: int) -> list[frozenset]:
    # Stable sort: ties broken by column order, which is fixed across runs.
    idx = np.argsort(-np.abs(A), axis=1, kind="stable")[:, :k]
    return [frozenset(r) for r in idx]


def pairwise_topk_overlap(runs: list[np.ndarray], k: int) -> np.ndarray:
    """Mean over run pairs of |top-k(a) ∩ top-k(b)| / k, per case."""
    sets = [topk_sets(A, k) for A in runs]
    n_cases = runs[0].shape[0]
    acc = np.zeros(n_cases); n = 0
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            acc += np.array([len(a & b) / k for a, b in zip(sets[i], sets[j])]); n += 1
    return acc / n


def sign_agreement(runs: list[np.ndarray], near_zero: float) -> np.ndarray:
    """Per case: over variables that are non-negligible (|a| >= near_zero) in at least
    one run, the mean fraction of runs agreeing with the majority sign; runs where the
    attribution is negligible count as non-agreeing ("no stable direction")."""
    S = np.stack(runs)                    # runs x cases x vars
    pos = (S >= near_zero).sum(0); neg = (S <= -near_zero).sum(0)
    active = (pos + neg) > 0
    frac = np.maximum(pos, neg) / S.shape[0]
    out = np.where(active, frac, np.nan)
    return np.nanmean(out, axis=1)


def sign_conflict(runs: list[np.ndarray], near_zero: float) -> np.ndarray:
    """Per case: share of non-negligible variables that are clearly positive in some runs
    and clearly negative in others (a direction reversal, as opposed to dropping out)."""
    S = np.stack(runs)
    pos = (S >= near_zero).any(0); neg = (S <= -near_zero).any(0)
    active = pos | neg
    return np.where(active.any(1), (pos & neg).sum(1) / np.maximum(active.sum(1), 1), np.nan)


def pairwise_rel_l1(runs: list[np.ndarray]) -> np.ndarray:
    """Mean over run pairs of ||a-b||_1 / (0.5(||a||_1+||b||_1)), per case (0 = identical)."""
    n_cases = runs[0].shape[0]
    acc = np.zeros(n_cases); n = 0
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i], runs[j]
            den = 0.5 * (np.abs(a).sum(1) + np.abs(b).sum(1))
            acc += np.abs(a - b).sum(1) / np.where(den > 0, den, np.nan); n += 1
    return acc / n


def pairwise_jaccard(sets_runs: list[list[frozenset]]) -> np.ndarray:
    n_cases = len(sets_runs[0])
    acc = np.zeros(n_cases); n = 0
    for i in range(len(sets_runs)):
        for j in range(i + 1, len(sets_runs)):
            acc += np.array([len(a & b) / len(a | b) if (a | b) else 1.0
                             for a, b in zip(sets_runs[i], sets_runs[j])]); n += 1
    return acc / n
