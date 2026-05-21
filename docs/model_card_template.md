<!--
  Copyright 2026 Srikumar Krishnamoorthy

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

# Model Card: {{ model_name }}

> **Template** — fill in all `{{ ... }}` placeholders before publication.
> Generated cards may also be produced programmatically via
> `hugiml.governance.generate_model_card()`.

---

## Overview

| Field               | Value |
|---------------------|-------|
| **Model name**      | {{ model_name }} |
| **Version**         | {{ model_version }} |
| **Algorithm**       | HUGIMLClassifierNative (HUG-IML, IEEE Access 2024) |
| **License**         | Apache 2.0 |
| **Owner**           | {{ owner }} |
| **Date**            | {{ date }} |
| **Contact**         | {{ contact }} |

---

## Intended Use

### Primary use case
{{ primary_use_case }}

### Out-of-scope uses
{{ out_of_scope }}

---

## Training Data

| Field                        | Value |
|------------------------------|-------|
| **Dataset name**             | {{ training_dataset_name }} |
| **Dataset description**      | {{ training_dataset_description }} |
| **Number of samples**        | {{ n_samples }} |
| **Number of features**       | {{ n_features }} |
| **Target variable**          | {{ target_variable }} |
| **Class distribution**       | {{ class_distribution }} |
| **Data collection period**   | {{ data_collection_period }} |
| **Known biases / caveats**   | {{ known_biases }} |

---

## Evaluation

### Test data
{{ test_data_description }}

### Metrics

| Metric          | Value |
|-----------------|-------|
| Accuracy        | {{ accuracy }} |
| ROC-AUC         | {{ roc_auc }} |
| Brier score     | {{ brier_score }} |
| ECE             | {{ ece }} |
| F1 (macro)      | {{ f1_macro }} |

### Calibration

{{ calibration_notes }}

---

## HUG-IML Specifics

| Field                      | Value |
|----------------------------|-------|
| **Number of HUG patterns** | {{ n_patterns }} |
| **Top patterns**           | {{ top_patterns }} |
| **Pattern mine params**    | B={{ B }}, L={{ L }}, G={{ G }} |

---

## Fairness & Bias

{{ fairness_notes }}

---

## Limitations

{{ limitations }}

---

## Governance

| Field                    | Value |
|--------------------------|-------|
| **Model ID**             | {{ model_id }} |
| **Approval status**      | {{ approval_status }} |
| **Review date**          | {{ review_date }} |
| **Approved by**          | {{ approved_by }} |

---


## References

Krishnamoorthy, S. (2024). Interpretable Classifier Models for Decision
Support Using High Utility Gain Patterns. *IEEE Access*, 12, 126088–126107.
DOI: [10.1109/ACCESS.2024.3455563](https://doi.org/10.1109/ACCESS.2024.3455563)
