"""Contextual help for public dashboard controls and evidence."""

import dash_bootstrap_components as dbc
from dash import html

CONTROL_HELP = {
    "Dataset": "Choose a synthetic example. Credit risk has 800 rows, customer churn 900, and claims review 720. No personal data is uploaded.",
    "Pattern budget": "topK controls the pattern mining budget. A larger value can provide more detail but may take longer. The number retained can differ from this budget.",
    "Pattern order": "L=1 searches individual feature conditions. L=2 also allows pairs of conditions. Larger orders can describe interactions but are harder to interpret.",
    "Estimator selection": "Automatic fits the bounded LR and RPTE candidates and selects the higher mean three-fold validation ROC-AUC. Manual selection bypasses this comparison.",
    "3-fold validation": "The training partition is split into three folds. Each fold is predicted using a model trained on the other two. Selection scores are not an independent test estimate.",
}
METRIC_HELP = {
    "Removed items": "Number of raw inputs or mined patterns removed in this comparison. Other representations may still carry related information.",
    "Holdout AUC change": "Pruned minus original ROC-AUC on the same reserved holdout. Positive means better ranking in this comparison; repeated exploration is not a final independent test.",
    "Changed predictions": "Number of holdout examples whose predicted class changed after pruning, using each model's default decision threshold.",
    "ROC-AUC": "How well the model ranks positive cases above negative cases across thresholds. 1 is perfect ranking; 0.5 is chance-level ranking. This does not measure probability calibration.",
    "Mean CV ROC-AUC": "Mean ranking score across the three validation folds. Automatic estimator selection uses this score; it is not an independent test score.",
    "Best CV ROC-AUC": "Mean three-fold validation ROC-AUC of the selected model. Automatic selection compares candidates using this score.",
    "Brier": "Mean squared difference between predicted probabilities and binary outcomes. Lower is better; 0 is perfect. Interpret alongside the positive rate and calibration plot.",
    "F1": "Harmonic mean of precision and recall at the displayed decision threshold. Higher values indicate a better balance between the two.",
    "Accuracy": "Share of cases classified correctly at the displayed threshold. It can be misleading when one outcome is much more common.",
    "Positive Rate": "Share of observed examples with the positive target label. This describes the dataset, not model accuracy.",
    "Rows Reviewed": "Number of training examples reviewed. A separate 25% partition is reserved for pruning comparisons.",
    "Model Features": "Number of input features after column roles and exclusions are applied. Derived patterns are additional representation features.",
    "Metric": "The scoring measure used for validation. ROC-AUC measures ranking across decision thresholds.",
    "Rows": "Number of rows in this validation results table; for this demo these are the three validation folds.",
}
COLUMN_HELP = {
    "fold": "Validation split number. Each fold is evaluated using training data from the other folds.",
    "validation_roc_auc": METRIC_HELP["Mean CV ROC-AUC"],
    "score": "Score for this validation result. Refer to the metric label for its definition.",
    "threshold": "Predicted probabilities at or above this cutoff are assigned the positive class.",
    "precision": "Among predicted positive cases, the fraction that are actually positive.",
    "recall": "Among actual positive cases, the fraction identified by the model.",
    "specificity": "Among actual negative cases, the fraction identified as negative.",
    "support": "Frequency or coverage of a pattern, using the units shown in this table.",
    "status": "Whether training and evaluation completed for this result.",
    "topK": CONTROL_HELP["Pattern budget"],
    "L": CONTROL_HELP["Pattern order"],
}
CHART_HELP = {
    "Calibration": "Compare predicted probabilities with observed outcome rates. Points near the diagonal indicate better calibration; small groups can be noisy.",
    "Score distribution by class": "Compare predicted probabilities for the observed classes. Greater separation indicates stronger discrimination. Overlap shows ambiguous cases.",
    "Threshold sweep": "See how precision, recall, and specificity change as the decision cutoff moves. A suitable cutoff depends on the costs of missed and false positive cases.",
}


def help_label(label, message, key):
    return html.Span(
        [
            html.Span(label),
            html.Button(
                "?",
                id=key,
                type="button",
                className="help-trigger",
                **{"aria-label": f"About {label}", "aria-describedby": key + "-tip"},
            ),
            dbc.Tooltip(
                message,
                id=key + "-tip",
                target=key,
                placement="auto",
                trigger="hover focus",
                autohide=False,
                className="studio-tooltip",
            ),
        ],
        className="help-label",
    )


def add_guidance(pages):
    """Attach help to serialized presentation components without changing their values."""
    import json

    from plotly.utils import PlotlyJSONEncoder

    counter = 0

    def label(text, message):
        nonlocal counter
        counter += 1
        return json.loads(
            json.dumps(help_label(text, message, f"evidence-help-{counter}"), cls=PlotlyJSONEncoder)
        )

    def walk(node):
        if isinstance(node, list):
            return [walk(value) for value in node]
        if not isinstance(node, dict):
            return node
        node = {key: walk(value) for key, value in node.items()}
        props = node.get("props", {})
        if props.get("className") == "mc-l" and isinstance(props.get("children"), str):
            title = props["children"]
            if title in METRIC_HELP:
                props["children"] = label(title, METRIC_HELP[title])
        if node.get("type") == "DataTable":
            props["tooltip_header"] = {
                column["id"]: COLUMN_HELP[column["id"]]
                for column in props.get("columns", [])
                if column["id"] in COLUMN_HELP
            }
            props["tooltip_delay"] = 250
            props["tooltip_duration"] = None
        if node.get("type") == "Graph":
            title = props.get("figure", {}).get("layout", {}).get("title", {})
            title = title.get("text", "") if isinstance(title, dict) else title
            if title in CHART_HELP:
                return {
                    "type": "Div",
                    "namespace": "dash_html_components",
                    "props": {
                        "className": "guided-chart",
                        "children": [label("Chart guide", CHART_HELP[title]), node],
                    },
                }
        return node

    return walk(pages)
