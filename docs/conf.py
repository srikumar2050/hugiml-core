"""Sphinx configuration for the hugiml-core documentation."""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "hugiml-core"
author = "Srikumar Krishnamoorthy"
copyright = f"{datetime.now().year}, {author}"

init_text = (SRC / "hugiml" / "__init__.py").read_text(encoding="utf-8")
release_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init_text)
release = release_match.group(1) if release_match else ""
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "myst_parser",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "exclude-members": "set_fit_request,set_score_request,set_predict_request,set_transform_request,set_partial_fit_request",
}
autodoc_mock_imports = [
    "_hugiml_core",
    "plotly",
    "shap",
    "mlflow",
    "opentelemetry",
    "prometheus_client",
    "imblearn",
    "xgboost",
    "interpret",
    "pygam",
    "rulefit",
    "streamlit",
    "altair",
    "jinja2",
    "markdown",
    "openpyxl",
    "lightgbm",
    "imodels",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "images/header-hugiml.png"
html_favicon = None
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "style_external_links": True,
}

todo_include_todos = False
suppress_warnings = ["ref.python", "ref.term", "ref.ref", "toc.not_included", "intersphinx"]
