"""Data audit, modelling population, design matrix and frozen splits."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd

from hxai.common import MODELS, TABLES, RunLog, dump_json, load_config, RESULTS
from hxai.data import audit, design_matrix, load_raw, make_population, make_splits

cfg = load_config()
log = RunLog("01_audit")
raw = load_raw(cfg)
a = audit(raw, cfg)
pd.DataFrame(a.pop("per_variable")).to_csv(TABLES / "audit_per_variable.csv", index=False)

pop = make_population(raw, cfg)
X, fmap = design_matrix(pop.X_raw, cfg)
splits = make_splits(pop, cfg)

# Category-level descriptive counts for the two coded fields (after exclusion).
cat_rows = []
for var in cfg["data"]["categorical"]:
    vc = pop.X_raw[var].value_counts().sort_index()
    for code, n in vc.items():
        cat_rows.append({"variable": var, "code": int(code), "n": int(n),
                         "bad_rate": float(pop.y[pop.X_raw[var].to_numpy() == code].mean())})
pd.DataFrame(cat_rows).to_csv(TABLES / "audit_categorical_codes.csv", index=False)

# Cross-tab used in the audit discussion: MaxDelqEver code vs MSinceMostRecentDelq special state.
state = pop.X_raw.MSinceMostRecentDelq.map({-7: "-7", -8: "-8"}).fillna(">=0")
pd.crosstab(state, pop.X_raw.MaxDelqEver).to_csv(TABLES / "audit_maxdelqever_vs_msincedelq.csv")

dev, test = splits["dev"], splits["test"]
pop_summary = {
    "n_population": int(len(pop.y)),
    "n_features_design": int(X.shape[1]),
    "n_original_variables": int(fmap.variable.nunique()),
    "excluded": pop.excluded,
    "bad_rate_population": float(pop.y.mean()),
    "n_dev": int(len(dev)), "n_test": int(len(test)),
    "bad_rate_dev": float(pop.y[dev].mean()), "bad_rate_test": float(pop.y[test].mean()),
    "groups_with_more_than_one_row": int((np.bincount(pop.groups) > 1).sum()),
    "special_state_rows_excluding_all9": {
        c: int(X[c].sum()) for c in fmap.loc[fmap.kind == "special", "column"]},
}
# Every group must sit entirely on one side of each split.
for name, (a_idx, b_idx) in {"dev/test": (dev, test)}.items():
    assert not set(pop.groups[a_idx]) & set(pop.groups[b_idx]), name
for k, (tr, va) in enumerate(splits["cv"]):
    assert not set(pop.groups[tr]) & set(pop.groups[va])
    assert set(tr) | set(va) == set(dev)

dump_json({"raw_audit": a, "population": pop_summary}, RESULTS / "audit.json")
fmap.to_csv(TABLES / "feature_map.csv", index=False)
joblib.dump({"pop": pop, "X": X, "fmap": fmap, "splits": splits}, MODELS / "data_bundle.joblib")
log.note(f"population n={len(pop.y)}, design columns={X.shape[1]}, dev={len(dev)}, test={len(test)}")
log.save(population=pop_summary)
