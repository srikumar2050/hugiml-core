# Complexity units

HUGIML exposes three complementary complexity measures through one generic interface.

## Model units

A coarse count of active fitted components in the complete model.

- Linear models: active fitted terms.
- Decision trees and ensembles: active terminal leaves.
- RuleFit: active linear terms and active rules.
- EBM: active additive terms.
- HUGIML RPTE: active terminal leaves plus active direct terms.

## Model inspection units

The expanded burden of inspecting the complete fitted model.

- HUGIML linear terms are expanded into their original source elements.
- Tree models sum the conditions on every active root-to-leaf path.
- RuleFit sums direct linear terms and the literals in every active rule.
- EBM counts active term-score cells.
- HUGIML RPTE sums every active terminal path and active direct source terms.

`get_complexity(model)` returns this measure by default.

## Instance inspection units

The expanded burden of inspecting one prediction.

- A tree contributes only the path reached by that instance.
- An ensemble contributes a reached path when its reached terminal output is active.
- HUGIML RPTE adds all active direct terms to the reached paths.
- Direct linear terms count for every instance.
- HUGIML patterns, augmented pairs, and RuleFit rules count only when they carry non-zero evidence for that instance.
- EBM terms count when their row-specific score contribution is non-zero.
- Intercepts are excluded.

For an RPTE model and row `x`:

```text
instance inspection units(x)
    = sum of expanded conditions on reached active leaf paths
    + expanded active direct terms
```

For the HUGIML linear branch, active original/direct terms count for every row.
Active pattern and augmented-pair terms count only when their transformed value
is non-zero for that row, and each counted term is expanded into its source elements.

`get_instance_inspection_units(model, X)` returns one integer count per row.
`get_complexity_report(model, X=X)` adds the mean, sample standard deviation,
standard error, and two-sided Student-t confidence interval.

```python
from hugiml import (
    get_complexity,
    get_complexity_report,
    get_instance_inspection_units,
)

model_inspection_units = get_complexity(model)
model_units = get_complexity(model, "model units")
instance_counts = get_instance_inspection_units(model, X_test)
instance_mean = get_complexity(model, "instance inspection units", X=X_test)
report = get_complexity_report(model, X=X_test, confidence_level=0.95)
```

For nested cross-validation, instance counts are evaluated only on outer-test
folds. Fold sufficient statistics are pooled so each dataset row appears once
in the dataset mean and confidence interval. Cross-dataset summaries give each
dataset equal weight and compute a separate confidence interval across dataset
means.
