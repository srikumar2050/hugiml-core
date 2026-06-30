"""Source-checkout wrapper for the packaged HUGIML NLP Streamlit UI.

From a checkout, run:

    PYTHONPATH=src streamlit run LLM/ui/hugiml_llm_chat.py

After installation with the optional extra, the simpler command is:

    hugiml-llm
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hugiml.llm.ui_app import main  # noqa: E402

if __name__ == "__main__":
    main()
