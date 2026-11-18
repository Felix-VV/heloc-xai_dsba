"""Shared configuration, paths and run-metadata helpers."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
LOGS = RESULTS / "logs"
MODELS = RESULTS / "models"


def load_config(path: Path | None = None) -> dict:
    path = path or ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def expand_grid_spec(spec):
    """`{logspace: [a, b, n]}` -> numpy array; plain lists pass through."""
    if isinstance(spec, dict) and "logspace" in spec:
        a, b, n = spec["logspace"]
        return np.logspace(a, b, int(n))
    return np.asarray(spec)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs() -> None:
    for d in (RESULTS, TABLES, FIGURES, LOGS, MODELS):
        d.mkdir(parents=True, exist_ok=True)


def versions() -> dict:
    import shap
    import sklearn
    import xgboost
    import pandas
    import scipy

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
    }


class RunLog:
    """Collects a JSON run record (timings, versions, notes) for one script."""

    def __init__(self, name: str):
        ensure_dirs()
        self.name = name
        self.t0 = time.time()
        self.record = {
            "script": name,
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": versions(),
            "notes": [],
        }

    def note(self, msg: str) -> None:
        print(f"[{self.name}] {msg}", flush=True)
        self.record["notes"].append(msg)

    def save(self, **extra) -> None:
        self.record.update(extra)
        self.record["elapsed_seconds"] = round(time.time() - self.t0, 1)
        self.record["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(LOGS / f"{self.name}.json", "w") as f:
            json.dump(self.record, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dump_json(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)
