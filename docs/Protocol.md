# Scientific protocol

## Population and target

Use the exact raw HELOC CSV identified in data/README.md. Bad=1. Exclude records
with all 23 predictors equal to -9; preserve other partial special-code records.
Keep identical records in one main split. Conclusions concern the retained bureau
population, not applicants with no bureau information. Target-window and sampling
uncertainties are disclosed.

## Main experiment

- Fixed group-aware stratified 80/20 development/test split; shared five development folds.
- Fit learned preprocessing only on training portions. Use semantic special-state
  indicators and dictionary-based categorical encoding, with the same information
  available to all model classes.
- Tune 20 L1 penalties, 80 depth-limited CART configurations and 60 XGBoost
  configurations by CV Brier. Early stopping uses training-fold internal data only.
- Repeat selection on ten shared 75/25 development splits with inner three-fold CV.
- Report AUC, Brier, log-loss, reliability and paired differences. Conditional test
  percentile intervals use 2,000 paired resamples. No equivalence margin is asserted.
- Select cases by reproducible rules; cases illustrate mechanisms, not general efficacy.

## Explanation and sensitivity

Use log-odds contributions and a fixed 200-development-row reference. Sum encoded
contributions to original variables for display; this is not group Shapley.
Thirty fixed-hyperparameter refits change both the training subset and seed. A
separate XGBoost seed-only arm and fixed-model background redraws distinguish other
sources. Changing Good/Bad reference populations changes explanation semantics.
No verified Rashomon set or user-understanding outcome is claimed.

## Exploratory score removal

After inspecting main results, remove only ExternalRiskEstimate and its indicator.
Keep the population and every split fixed. Retune each model using the same original
search spaces and criteria, including all ten development repetitions. Differences
are full minus no-score. Compare aggregate metrics and individual probability
changes. This tests predictive dependence under refitting, not a causal intervention
on customers or a guarantee that all upstream opacity has been removed.

## Reporting

Results must come from saved predictions and actual runs. Avoid interpreting an
interval containing zero as equivalence, additivity as complete faithfulness or
causality, transparency as fairness, or internal resampling as external validation.
The concise report targets the course's approximately-five-page requirement;
complete notes retain additional details. Slides remain a separate team deliverable.
