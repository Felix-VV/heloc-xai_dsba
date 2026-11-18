# HELOC: interpretable models and explained black boxes

A credit-risk case study comparing L1 logistic regression, depth-limited CART and
XGBoost, with same-customer explanations, retraining/reference sensitivity, and an
exploratory removal of `ExternalRiskEstimate`. This is a comparison and audit
workflow using established methods, not a new learning algorithm.

## Main results

| Model | Full-input test AUC | No-score test AUC |
|---|---:|---:|
| L1 logistic | 0.7921 | 0.7910 |
| CART | 0.7585 | 0.7543 |
| XGBoost | 0.7956 | 0.7943 |

The two stronger models have close observed performance in both conditions; this
is not a formal equivalence result. Removing the score changes CART probabilities
by more than 0.10 for 46.8% of test customers, despite modest aggregate changes.
The score-removal analysis was specified after inspecting the main results and is
exploratory. One historical dataset does not establish causality, fairness, human
understanding, or external deployment validity.

## Setup

Tested with Python 3.13.7 on macOS/Apple Silicon. Exact Python package versions are
in `requirements-lock.txt`; `requirements.txt` lists top-level dependencies.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

Or use `bash setup.sh`. The project-local Python launcher limits numerical threads,
uses writable local plotting caches, and on macOS exposes the OpenMP runtime bundled
in the installed scikit-learn wheel to XGBoost. It does not modify global libraries.
Other operating systems may need their standard XGBoost runtime prerequisites.

For PDF compilation, install Tectonic or latexmk. On macOS the helper below fetches
an official, pinned portable Tectonic release into this project:

```bash
python3 tools/install_tectonic.py
```

The first build downloads TeX resources into `.cache/tectonic`. Internet is needed
for dependency installation, initial data retrieval and uncached TeX resources.

## Run

```bash
bash run_all.sh
```

This retrieves and hash-checks the data, audits and encodes it, reruns model selection,
ten repeated development splits, case explanations, stability analyses and the retuned
score ablation, then generates figures, tables, numerical checks and both reports.
The raw data and fitted models are deliberately excluded from this repository.
Default model tuning uses two worker processes.

```bash
bash run_all.sh --report-only
```

This works from the included saved outputs without raw data or fitted models.
It regenerates figures, checks saved-output consistency and compiles both reports.
It does not refit or revalidate model binaries. A full run additionally executes
`scripts/06_checks.py` and `scripts/09_extended_checks.py` for model-based checks.
`--resume-ablation` can reuse repeat checkpoints only when their signature matches.
`python3 tools/run_pipeline.py --dry-run` prints stage order without execution.

## Layout

- `src/hxai/`: data, model, metrics and explanation implementations
- `scripts/`: experiment, output generation and validation stages
- `config.yaml`: common splits, seeds, grids and explanation settings
- `data/README.md`: exact data sources, checksum and dictionary limitations
- `docs/Protocol.md`: scientific protocol and scope of inference
- `results/*.json`: saved numerical evidence
- `results/tables/`: predictions, searches, coefficients, cases and repeated results
- `results/figures/`: seven figure groups in PDF, PNG and SVG
- `report/submission.tex`: concise submission report source
- `report/report.tex`: complete research-notes source
- `report/refs.bib`: references
- `tools/`: isolated execution and report build helpers

PDFs are generated as `output/pdf/Report.pdf` and `output/pdf/Research_Notes.pdf`.
In the local handoff, copies and editable source are grouped in `../01_Research/`;
that sibling folder is not required to run this repository. Upload this `02_Code`
folder as the repository root, then add the actual repository URL to the team submission.

## Research boundaries

The 20% test set is fixed; preprocessing and tuning use training portions only.
AUC measures ranking; Brier/log-loss assess overall probability quality, not pure
calibration. Test bootstrap intervals condition on fitted models. Repeated development
splits overlap. Main refits change both data subset and seed; a separate XGBoost arm
changes seed alone. Reference-population swaps change the question being explained.
TreeSHAP reconstruction is not a causal or safety guarantee. Sensitive attributes,
user studies, external validation and a lending-cost model are not available.

## Verification evidence

`results/validation.json` records the final numerical, model and document checks.
The code folder also passed report-only reconstruction in isolation, with no raw
data, model binaries or sibling research folder. That check used an existing
Python environment and cached TeX resources; it was not a fresh dependency-install
test. Original experiment execution records retain their timestamps in `results/logs/`.
