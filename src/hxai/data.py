"""Loading, auditing and encoding the FICO HELOC data.

The design matrix is a row-wise, parameter-free transformation of the raw
record (no statistics are estimated here), so it can be built once for all
rows without leaking information across splits. Anything estimated from data
(imputation medians, scaling) lives inside the model pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .common import ROOT, sha256_of

SPECIAL_CODES = (-7, -8, -9)
SPECIAL_MEANING = {
    -9: "No Bureau Record or No Investigation",
    -8: "No Usable/Valid Trades or Inquiries",
    -7: "Condition not Met (e.g. No Inquiries, No Delinquencies)",
}


def load_raw(cfg: dict) -> pd.DataFrame:
    path = ROOT / cfg["data"]["raw_path"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/README.md for how to obtain heloc_dataset_v1.csv."
        )
    digest = sha256_of(path)
    if digest != cfg["data"]["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: got {digest}. The pipeline was run on the file "
            f"with hash {cfg['data']['sha256']}; results may differ for another version."
        )
    return pd.read_csv(path)


def predictor_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    return [c for c in df.columns if c != cfg["data"]["target"]]


def audit(df: pd.DataFrame, cfg: dict) -> dict:
    """Descriptive audit of the raw file; uses the label only for reporting class balance."""
    tgt = cfg["data"]["target"]
    X = df[predictor_columns(df, cfg)]
    all9 = (X == -9).all(axis=1)
    out = {
        "n_rows": int(len(df)),
        "n_predictors": int(X.shape[1]),
        "label_counts": df[tgt].value_counts().to_dict(),
        "all_minus9_rows": int(all9.sum()),
        "all_minus9_label_counts": df.loc[all9, tgt].value_counts().to_dict(),
        "rows_with_any_minus9_not_all": int(((X == -9).any(axis=1) & ~all9).sum()),
        "exact_duplicate_rows_total": int(df.duplicated().sum()),
        "exact_duplicate_rows_excluding_all_minus9": int(df[~all9].duplicated().sum()),
        "unexpected_negative_values": {},
    }
    per_var = []
    for c in X.columns:
        v = X[c]
        valid = v[v >= 0]
        other_neg = sorted(set(v[v < 0].unique()) - set(SPECIAL_CODES))
        if other_neg:
            out["unexpected_negative_values"][c] = other_neg
        row = {"variable": c, "n_unique_valid": int(valid.nunique()),
               "min_valid": float(valid.min()), "median_valid": float(valid.median()),
               "p99_valid": float(valid.quantile(0.99)), "max_valid": float(valid.max())}
        for code in SPECIAL_CODES:
            row[f"n_{code}"] = int((v == code).sum())
            row[f"n_{code}_excl_all9"] = int((v[~all9] == code).sum())
        per_var.append(row)
    out["per_variable"] = per_var
    # Duplicates after exclusion: are labels consistent within identical predictor vectors?
    d = df[~all9]
    grp = d.groupby(list(X.columns))[tgt].nunique()
    out["duplicate_groups_with_conflicting_labels"] = int((grp > 1).sum())
    dup_mask = d.duplicated(keep=False)
    out["duplicate_row_indices_excluding_all_minus9"] = d.index[dup_mask].tolist()
    return out


@dataclass
class Population:
    X_raw: pd.DataFrame        # raw predictor codes, modelling population only
    y: np.ndarray              # 1 = Bad
    row_id: np.ndarray         # 0-based row index in the raw file
    groups: np.ndarray         # identical predictor+label rows share a group (kept in one split)
    excluded: dict


def make_population(df: pd.DataFrame, cfg: dict) -> Population:
    tgt = cfg["data"]["target"]
    cols = predictor_columns(df, cfg)
    X = df[cols]
    keep = np.ones(len(df), dtype=bool)
    excluded = {}
    if cfg["data"]["exclude_all_special_9"]:
        all9 = (X == -9).all(axis=1).to_numpy()
        keep &= ~all9
        excluded["all_predictors_minus9"] = int(all9.sum())
    sub = df.loc[keep].copy()
    y = (sub[tgt] == cfg["data"]["positive_label"]).astype(int).to_numpy()
    # Exact duplicates cannot be confirmed as the same applicant (no ID), but they are
    # grouped so that a copy never sits on both sides of a split.
    key = pd.util.hash_pandas_object(sub[cols + [tgt]], index=False).to_numpy()
    _, groups = np.unique(key, return_inverse=True)
    return Population(X_raw=sub[cols].reset_index(drop=True), y=y,
                      row_id=sub.index.to_numpy(), groups=groups, excluded=excluded)


def design_matrix(X_raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode raw codes into model inputs shared by all three model families.

    Returns (X, feature_map). Special codes become NaN plus explicit 0/1 state
    indicators; coded categorical fields become dummies (reference level dropped).
    """
    dcfg = cfg["data"]
    cat_cfg = dcfg["categorical"]
    ind_cfg = dcfg["special_indicators"]
    cols, fmap = {}, []

    for var in X_raw.columns:
        v = X_raw[var]
        if var in cat_cfg:
            spec = cat_cfg[var]
            known = {code for codes in spec["levels"].values() for code in codes}
            bad = sorted(set(v.unique()) - known)
            if bad:
                raise ValueError(f"{var}: codes {bad} not in data dictionary levels")
            for level, codes in spec["levels"].items():
                # Dictionary levels that never occur would only add constant columns.
                if level == spec["reference"] or not v.isin(codes).any():
                    continue
                name = f"{var}={level}"
                cols[name] = v.isin(codes).astype(float)
                fmap.append((name, var, "category", level))
            continue
        num = v.astype(float).where(~v.isin(SPECIAL_CODES))
        cols[var] = num
        fmap.append((var, var, "numeric", ""))
        for code in ind_cfg.get(var, []):
            name = f"{var}=={code}"
            cols[name] = (v == code).astype(float)
            fmap.append((name, var, "special", str(code)))
        undeclared = sorted(set(v[v.isin(SPECIAL_CODES)].unique()) - set(ind_cfg.get(var, [])))
        if undeclared:
            raise ValueError(f"{var}: special codes {undeclared} present but no indicator declared")

    X = pd.DataFrame(cols)
    feature_map = pd.DataFrame(fmap, columns=["column", "variable", "kind", "level"])
    return X, feature_map


# ---------------------------------------------------------------- splits

def _first_fold(y, groups, n_splits, seed):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest, held = next(iter(sgkf.split(np.zeros(len(y)), y, groups)))
    return np.sort(rest), np.sort(held)


def make_splits(pop: Population, cfg: dict) -> dict:
    """All index sets are positions in the population arrays."""
    seed = cfg["seed"]
    scfg = cfg["split"]
    n_out = int(round(1 / scfg["test_size"]))
    dev, test = _first_fold(pop.y, pop.groups, n_out, seed)

    def folds(idx, k, s):
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=s)
        return [(idx[a], idx[b]) for a, b in sgkf.split(np.zeros(len(idx)), pop.y[idx], pop.groups[idx])]

    cv = folds(dev, scfg["cv_folds"], seed + 1)
    rep = []
    r = scfg["repeated"]
    n_hold = int(round(1 / r["holdout_size"]))
    for i in range(r["n_repeats"]):
        a, b = _first_fold(pop.y[dev], pop.groups[dev], n_hold, seed + 100 + i)
        tr, ho = dev[a], dev[b]
        rep.append({"train": tr, "holdout": ho, "inner": folds(tr, r["inner_folds"], seed + 200 + i)})
    return {"dev": dev, "test": test, "cv": cv, "repeated": rep}
