# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

project = "hugiml-core"
author = "Srikumar Krishnamoorthy"
release = "1.1.4"
version = "1.1.4"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_typehints = "description"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
html_title = "hugiml-core 1.1.4"

suppress_warnings = ["ref.term", "ref.ref"]
