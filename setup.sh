#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
python3 tools/run_python.py -c 'import sklearn, xgboost, shap; print("Research environment ready:", sklearn.__version__, xgboost.__version__, shap.__version__)'
