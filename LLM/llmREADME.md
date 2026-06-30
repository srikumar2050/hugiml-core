# Optional HUGIML Natural-Language Add-on

This folder contains additive assets for a local natural-language interface over existing HUGIML APIs. Ollama remains optional and external; normal `import hugiml` does not require Ollama, Streamlit, or Plotly.

## Install and launch

After the package is installed with the optional extra:

```bash
pip install "hugiml-core[llm]"
hugiml-llm
```

The default `hugiml-llm` command launches the Streamlit natural-language workbench, parallel to `hugiml-dashboard`.

Useful subcommands:

```bash
hugiml-llm status
hugiml-llm list-datasets
hugiml-llm chat --dataset churn_synthetic --no-llm
hugiml-llm ask "tune churn_synthetic and explain the strongest patterns" --dataset churn_synthetic
hugiml-llm demo-html --output LLM/examples/governance_qna_churn.html
```

From a source checkout, before reinstalling:

```bash
PYTHONPATH=src python -m hugiml.llm.cli status
PYTHONPATH=src python -m hugiml.llm.cli list-datasets --repo-root .
PYTHONPATH=src streamlit run LLM/ui/hugiml_llm_chat.py
```

## Scope

The interface supports:

- listing and describing datasets
- HUGIML-only model build and tuning
- prediction and tabular output generation
- grounded model and prediction interpretation
- controlled pattern pruning
- governance/model-card artifact generation
- HUGIML API and feature help

It refuses:

- source-code rewrites
- arbitrary Python/script generation
- shell commands or package installation
- Git operations
- baseline model runs such as XGBoost, LightGBM, random forest, EBM, RuleFit, or logistic regression

## Dataset handling

The workbench presents a merged catalog while keeping origins separate:

- packaged/sample datasets for zero-config first use
- benchmark datasets exposed through the existing benchmark catalog when running from a source checkout
- user-uploaded datasets registered only after the target column is explicitly selected

When installed from a wheel, user datasets are stored under `~/.hugiml/llm/datasets/user` unless `HUGIML_LLM_HOME` is set.


## In-app examples and commands

The Streamlit workbench now includes a visible **Try these questions** section in the Chat tab and a **Commands** panel in the sidebar. Users can click example prompts such as:

- Build a HUGIML model
- Tune this dataset for ROC AUC
- Generate a prediction table
- Explain the strongest patterns
- Prune low-support patterns
- Generate a governance report

The sidebar also shows copyable CLI commands for `hugiml-llm`, `status`, `list-datasets`, `chat`, `ask`, and `demo-html`, so users do not need to open this README to discover the main entry points.

## UI visuals

The dashboard includes:

- dataset source bars and a row/feature size map
- class-balance charts
- metric cards and metric profile bars
- confusion matrix heatmap
- feature/pattern influence charts
- prediction probability charts
- pruning before/after readouts
- governance artifact panels

## Ollama posture

Ollama is optional and external. The add-on detects whether an Ollama server and model are available, but it never starts Ollama and never downloads models automatically. For v1, use detect-and-instruct setup.
